from __future__ import annotations

from flask import Blueprint, jsonify, request
from typing import Any, Dict, List, Tuple
import os
import sys
import time
import signal
import subprocess
import pandas as pd

from tgTrax.utils import tui
import asyncio
import threading
from telethon import TelegramClient, errors as tg_errors
from tgTrax.core.tracker import CorrelationTracker
from tgTrax.core.analysis import TemporalAnalyzer


api_bp = Blueprint("api", __name__)

# Lightweight in-memory cache for heavy responses
_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL_SECONDS: float = 60.0

def _cache_get(key: str) -> Any | None:
    ent = _CACHE.get(key)
    if not ent:
        return None
    ts, val = ent
    if (time.time() - ts) > _CACHE_TTL_SECONDS:
        try:
            del _CACHE[key]
        except Exception:
            pass
        return None
    return val

def _cache_set(key: str, val: Any) -> None:
    _CACHE[key] = (time.time(), val)

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _state_file() -> str:
    runtime_dir = os.path.join(_project_root(), 'runtime')
    try:
        os.makedirs(runtime_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(runtime_dir, 'tracker_state.json')


def _parse_since_param() -> pd.Timestamp | None:
    since = (request.args.get("since") or "24h").strip().lower()
    now = pd.Timestamp.utcnow()
    if since in ("all", "0"):
        return None
    if since == 'start':
        try:
            import json
            with open(_state_file(), 'r') as f:
                state = json.load(f)
            started_at = state.get('started_at')
            if started_at:
                ts = pd.Timestamp(started_at)
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                return ts
        except Exception:
            pass
        # Fallback to 24h if no state
        return now - pd.Timedelta(hours=24)
    if since.endswith("h") and since[:-1].isdigit():
        return now - pd.Timedelta(hours=int(since[:-1]))
    if since.endswith("d") and since[:-1].isdigit():
        return now - pd.Timedelta(days=int(since[:-1]))
    return now - pd.Timedelta(hours=24)


def _fetch_activity_df() -> pd.DataFrame:
    # Create cache key based on data parameters
    since_str = request.args.get("since", "24h")
    data_key = f"activity_df:{since_str}"
    cached = _cache_get(data_key)
    if cached is not None:
        return cached
    
    target_users_str = os.getenv("TARGET_USERS", "")
    target_users = [u.strip() for u in target_users_str.split(",") if u.strip()]
    db_path = os.getenv("TGTRAX_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tgTrax.db"))
    if not target_users:
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT username FROM activity")
            target_users = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception as e:
            tui.tui_print_warning(f"API could not discover users from DB: {e}")
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        tracker = CorrelationTracker(target_usernames=target_users, db_path=db_path)
        df = tracker.get_activity_data()
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    cutoff = _parse_since_param()
    if cutoff is not None and not df.empty:
        df = df[df.index >= cutoff]
    
    # Cache the result for 5 minutes
    _cache_set(data_key, df)
    return df


def _get_request_fingerprint() -> str:
    """Generate fingerprint for all request parameters."""
    import hashlib
    params = {
        'since': request.args.get('since', ''),
        'tz': request.args.get('tz', ''),
        'period': request.args.get('period', '1min'),
        'method': request.args.get('method', 'spearman'),
        'seasonal_adjust': request.args.get('seasonal_adjust', 'false'),
        'ewma_alpha': request.args.get('ewma_alpha', ''),
        'max_lag_minutes': request.args.get('max_lag_minutes', '15'),
        'fdr_alpha': request.args.get('fdr_alpha', '0.05'),
        'corr_threshold': request.args.get('corr_threshold', '0.6'),
        'jacc_threshold': request.args.get('jacc_threshold', '0.18'),
        'resample_agg': request.args.get('resample_agg', 'max'),
        'fill_missing': request.args.get('fill_missing', 'zero'),
        'resid_include_global': request.args.get('resid_include_global', 'false'),
        'resid_prior_strength': request.args.get('resid_prior_strength', '24'),
        'resid_varstab': request.args.get('resid_varstab', 'true'),
    }
    return hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:12]

def _build_analyzer_from_request() -> Tuple[TemporalAnalyzer, pd.DataFrame]:
    # Check cache first
    fingerprint = _get_request_fingerprint()
    cache_key = f"analyzer:{fingerprint}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    
    df = _fetch_activity_df()
    # Timezone handling: allow explicit tz conversion
    tz = request.args.get("tz")
    if tz:
        try:
            if df.index.tz is None:
                df = df.tz_localize('UTC').tz_convert(tz)
            else:
                df = df.tz_convert(tz)
            # keep tz-aware index; downstream residualize_time respects tz if present
        except Exception:
            pass
    period = request.args.get("period", "1min")
    method_req = (request.args.get("method", "spearman") or "spearman").lower()
    seasonal_adjust = request.args.get("seasonal_adjust", "false").lower() == "true"
    ewma_alpha = request.args.get("ewma_alpha")
    ewma_alpha_val = float(ewma_alpha) if ewma_alpha is not None else None
    max_lag_minutes = int(request.args.get("max_lag_minutes", "15"))
    fdr_alpha = float(request.args.get("fdr_alpha", "0.05"))
    corr_thr = float(request.args.get("corr_threshold", "0.6"))
    jacc_thr = float(request.args.get("jacc_threshold", "0.18"))
    # resampling policies
    resample_agg = (request.args.get("resample_agg") or "max").lower()
    fill_missing = (request.args.get("fill_missing") or "zero").lower()
    # residualization defaults
    resid_global = (request.args.get("resid_include_global") or "false").lower() == "true"
    resid_prior = int(request.args.get("resid_prior_strength", "24"))
    resid_varstab = (request.args.get("resid_varstab") or "true").lower() == "true"

    # Only valid correlation methods for the analyzer
    corr_method = method_req if method_req in ("spearman", "pearson") else "spearman"
    # auto-tune period to satisfy min_L/min_nonzero if requested
    auto_tune = (request.args.get("auto_tune") or "false").lower() == "true"
    min_L_req = int(request.args.get("min_L", "0"))
    min_nnz_req = int(request.args.get("min_nonzero", "0"))
    if auto_tune and df is not None and not df.empty:
        candidates = [period] if period else []
        for s in ("1min", "2min", "3min", "5min", "10min"):
            if s not in candidates:
                candidates.append(s)
        chosen = None
        for p in candidates:
            try:
                df_num = df.astype(float).resample(p).max()
                L = len(df_num.index)
                nnz_ok = True
                if min_nnz_req > 0:
                    binf = df_num.astype(bool).astype(int)
                    for c in binf.columns:
                        if int(binf[c].sum()) < min_nnz_req:
                            nnz_ok = False; break
                if (min_L_req == 0 or L >= min_L_req) and nnz_ok:
                    chosen = p
                    break
            except Exception:
                continue
        if chosen:
            period = chosen
    analyzer = TemporalAnalyzer(
        df,
        resample_period=period,
        correlation_threshold=corr_thr,
        jaccard_threshold=jacc_thr,
        seasonal_adjust=seasonal_adjust,
        ewma_alpha=ewma_alpha_val,
        resample_agg=resample_agg,
        fill_missing=fill_missing,
        max_lag_minutes=max_lag_minutes,
        corr_method=corr_method,
        fdr_alpha=fdr_alpha,
        residual_include_global=resid_global,
        residual_prior_strength=resid_prior,
        residual_variance_stabilize=resid_varstab,
    )
    result = (analyzer, df)
    _cache_set(cache_key, result)  # Cache for 10 minutes
    return result


@api_bp.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@api_bp.get("/config")
def config_view() -> Any:
    users_env = os.getenv("TARGET_USERS", "")
    users_list = [u.strip() for u in users_env.split(",") if u.strip()]
    db_path = os.getenv("TGTRAX_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tgTrax.db"))
    # Try to include session path
    session_path = None
    try:
        _api_id, _api_hash, sess = _get_api_credentials()
        session_path = f"{sess}.session"
    except Exception:
        pass
    return jsonify({
        "target_users_env": users_list,
        "db_path": db_path,
        "session_path": session_path,
    })


@api_bp.get("/summary")
def summary() -> Any:
    # Lightweight summary (no cross-correlation)
    try:
        df = _fetch_activity_df()
        period = request.args.get("period", "1min")
        if df.empty:
            return jsonify({"summary": {"num_users": 0, "num_resampled_periods": 0, "avg_online_periods_per_user": 0, "total_duration_analyzed": "0"}, "users": []})
        users = list(df.columns)
        df_num = df.astype(float).resample(period).max().fillna(0)
        num_periods = len(df_num.index)
        if num_periods:
            start, end = df_num.index[0], df_num.index[-1]
            delta = end - start
            days = delta.days
            hours = (delta.seconds) // 3600
            minutes = (delta.seconds % 3600) // 60
            dur = f"{days}d {hours}h {minutes}m"
        else:
            dur = "0"
        online_counts = (df_num > 0).sum(axis=0)
        avg_online = float(online_counts.mean()) if not online_counts.empty else 0.0
        stats = {
            "num_users": len(users),
            "num_resampled_periods": num_periods,
            "avg_online_periods_per_user": avg_online,
            "total_duration_analyzed": dur,
        }
        return jsonify({"summary": stats, "users": users})
    except Exception as e:
        import traceback
        print(f"ERROR in /api/summary: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@api_bp.get("/matrices")
def matrices() -> Any:
    # Use full fingerprint for cache key
    fingerprint = _get_request_fingerprint()
    metric = (request.args.get("metric") or "spearman").lower()
    residualize = (request.args.get("residualize") or "false").lower() == "true"
    fdr_method = (request.args.get("fdr_method") or "by").lower()
    cache_key = f"matrices:{metric}:{residualize}:{fdr_method}:{fingerprint}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)
    
    analyzer, _ = _build_analyzer_from_request()

    def df_to_jsonable(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if df is None or df.empty:
            return {}
        return {r: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()} for r, row in df.to_dict(orient="index").items()}

    cache_key = f"matrices:{metric}:{residualize}:{request.args.get('since','')}:{request.args.get('period','')}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    if metric in ("spearman", "pearson"):
        mat = analyzer.compute_correlation_matrix(method="pearson" if metric == "pearson" else "spearman", residualize=residualize)
    elif metric in ("corr_pvals", "corr_qvals"):
        method = (request.args.get("method") or "spearman").lower()
        _c, p, q = analyzer.compute_correlation_pq(method=("pearson" if method == "pearson" else "spearman"), residualize=residualize, fdr_method=fdr_method)
        mat = p if metric == "corr_pvals" else q
    elif metric == "jaccard":
        mat = analyzer.get_jaccard_matrix()
    elif metric == "mcc":
        mat = analyzer.mcc_matrix()
    elif metric == "kappa":
        mat = analyzer.kappa_matrix()
    elif metric == "ochiai":
        mat = analyzer.ochiai_matrix()
    elif metric == "overlap":
        mat = analyzer.overlap_matrix()
    elif metric == "crosscorr_max":
        mat = analyzer.get_crosscorr_max()
    elif metric == "crosscorr_qvals":
        mat = analyzer.get_crosscorr_qvals()
    else:
        mat = pd.DataFrame()
    out = {"matrix": df_to_jsonable(mat), "meta": {"method": metric, "residualize": residualize, "since": request.args.get("since")}}
    _cache_set(cache_key, out)
    return jsonify(out)


@api_bp.get("/pairs/significant")
def pairs_significant() -> Any:
    # Add caching for this expensive endpoint
    fingerprint = _get_request_fingerprint()
    method = (request.args.get("method") or "xcorr").lower()
    residualize = (request.args.get("residualize") or "false").lower() == "true"
    cache_key = f"pairs_significant:{method}:{residualize}:{fingerprint}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)
    
    analyzer, _ = _build_analyzer_from_request()
    permute = (request.args.get("permute") or "false").lower() == "true"
    perms = max(10, min(200, int(request.args.get("perms", "100"))))
    jobs = max(0, int(request.args.get("jobs", "0")))
    bsz = max(1, min(64, int(request.args.get("batch_size", "16"))))
    tau_seconds = int(request.args.get("tau_seconds", "120"))
    fdr_method = (request.args.get("fdr_method") or "by").lower()
    te_bootstrap = (request.args.get("bootstrap") or "stationary").lower()
    te_quantize = (request.args.get("te_quantize") or "balanced").lower()
    te_block_p = float(request.args.get("block_p", "0.1"))
    qthr = request.args.get("q_threshold")
    qthr_val = float(qthr) if qthr is not None else None
    min_abs = float(request.args.get("min_abs_corr", "0.0"))
    top = int(request.args.get("top", "100"))
    # multi-scale agreement (optional)
    multi_scale = (request.args.get("multi_scale") or "false").lower() == "true"
    scales_raw = (request.args.get("scales") or "1min,3min,5min")
    scales = [s.strip() for s in scales_raw.split(',') if s.strip()]
    ms_min = max(1, int(request.args.get("ms_min_agree", "2")))
    # sample/stability gating
    min_L = int(request.args.get("min_L", "0"))
    min_nnz = int(request.args.get("min_nonzero", "0"))
    stable = (request.args.get("stable") or "false").lower() == "true"
    folds = max(2, min(6, int(request.args.get("folds", "2"))))
    stability_min = float(request.args.get("stability_min", "0.67"))
    stability_method = (request.args.get("stability_method") or "block").lower()

    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "method": method,
        "residualize": residualize,
        "permute": permute,
        "perms": perms,
        "jobs": (jobs or None),
        "tau_seconds": tau_seconds,
        "since": request.args.get("since"),
        "fdr_method": fdr_method,
        "pmin": (1.0 / (perms + 1.0)) if (permute or method == 'te') else None,
        "min_L": min_L,
        "min_nonzero": min_nnz,
        "stable": stable,
        "folds": folds,
        "stability_min": stability_min,
        "stability_method": stability_method,
    }

    # Precompute residuals if needed
    if residualize:
        try:
            analyzer.residualize_time()
        except Exception:
            pass

    if method == "xcorr":
        if multi_scale and len(scales) >= 2:
            # Iterate through scales and intersect pairs
            from collections import Counter
            agree_counter: Counter[tuple] = Counter()
            details: Dict[tuple, Dict[str, Any]] = {}
            df = _fetch_activity_df()
            for sc in scales:
                try:
                    a_sc = TemporalAnalyzer(
                        df,
                        resample_period=sc,
                        correlation_threshold=float(request.args.get("corr_threshold", "0.6")),
                        jaccard_threshold=float(request.args.get("jacc_threshold", "0.18")),
                        seasonal_adjust=(request.args.get("seasonal_adjust", "false").lower()=="true"),
                        ewma_alpha=(float(request.args.get("ewma_alpha")) if request.args.get("ewma_alpha") else None),
                        max_lag_minutes=int(request.args.get("max_lag_minutes", "15")),
                        corr_method=analyzer.corr_method,
                        fdr_alpha=float(request.args.get("fdr_alpha", "0.05")),
                    )
                except Exception:
                    continue
                if residualize:
                    try:
                        a_sc.residualize_time()
                    except Exception:
                        pass
                xr_sc, lag_sc, _pv_sc, q_sc = a_sc.compute_crosscorr_with_lag(method=a_sc.corr_method, residualize=residualize, fdr_method=fdr_method)
                users_sc = list(xr_sc.columns) if not xr_sc.empty else a_sc.user_list
                for i in range(len(users_sc)):
                    for j in range(i+1, len(users_sc)):
                        u, v = users_sc[i], users_sc[j]
                        try:
                            xr = float(xr_sc.at[u, v])
                            qv = float(q_sc.at[u, v]) if not q_sc.empty else float('nan')
                        except Exception:
                            xr, qv = float('nan'), float('nan')
                        cond_q = (qthr_val is not None) and (not pd.isna(qv)) and (qv <= qthr_val)
                        cond_r = (not pd.isna(xr)) and (abs(xr) >= min_abs)
                        if cond_q or cond_r:
                            key = (u, v)
                            agree_counter[key] += 1
                            if key not in details:
                                details[key] = {"xr": xr, "q": qv, "lag": (float(lag_sc.at[u, v]) if not lag_sc.empty else None)}
            for (u, v), c in agree_counter.items():
                if c >= ms_min:
                    d = details.get((u, v), {})
                    rows.append({
                        "u": u, "v": v, "score": d.get("xr"), "r": d.get("xr"),
                        "q": d.get("q"), "lag_seconds": d.get("lag"), "source": "xcorr",
                        "agree_scales": int(c), "scales": scales,
                    })
            rows = sorted(rows, key=lambda x: ((x.get("q", 1.0) if x.get("q") is not None else 1.0), -abs(x.get("score", 0.0) or 0.0)))[:top]
            return jsonify({"pairs": rows, "meta": {"method": method, "residualize": residualize, "multi_scale": True, "scales": scales, "ms_min_agree": ms_min }})
        # Compute xcorr matrices from the correct frame (residuals if requested)
        xr_mat, lag_mat, _pv_obs, q_mat_obs = analyzer.compute_crosscorr_with_lag(
            method=analyzer.corr_method, residualize=residualize, fdr_method=fdr_method
        )
        # If requested, compute permutation q-values (overwrites q_mat_obs)
        q_mat = q_mat_obs
        if permute:
            kw = {"perms": perms, "method": analyzer.corr_method, "use_residuals": residualize, "batch_size": bsz, "fdr_method": fdr_method}
            if jobs > 0:
                kw["n_jobs"] = jobs
            _p, q_mat = analyzer.crosscorr_pvals_max_lag_perm(**kw)
        jacc = analyzer.get_jaccard_matrix()
        corr = analyzer.compute_correlation_matrix(residualize=residualize)
        users = list(corr.columns) if not corr.empty else analyzer.user_list
        # sample stats and optional stability
        frame = analyzer.df_resid if (residualize and getattr(analyzer, 'df_resid', None) is not None) else analyzer.df_for_corr
        T_eff = len(frame.index) if frame is not None and not frame.empty else 0
        binf = analyzer._binary_frame()
        nnz = {u: int(binf[u].sum()) for u in users} if (binf is not None and not binf.empty) else {u: 0 for u in users}
        stab_mat = None
        if stable:
            stab_mat = analyzer.stability_matrix(method=stability_method, residualize=residualize, folds=folds, min_abs=min_abs)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u, v = users[i], users[j]
                try:
                    xr = float(xr_mat.at[u, v]) if not xr_mat.empty else float("nan")
                except Exception:
                    xr = float("nan")
                try:
                    lag_s = float(lag_mat.at[u, v]) if not lag_mat.empty else float("nan")
                except Exception:
                    lag_s = float("nan")
                try:
                    qv = float(q_mat.at[u, v]) if (q_mat is not None and not q_mat.empty) else float("nan")
                except Exception:
                    qv = float("nan")
                # Thresholds and gating
                cond_q = (qthr_val is not None) and (not pd.isna(qv)) and (qv <= qthr_val)
                cond_r = (not pd.isna(xr)) and (abs(xr) >= min_abs)
                if min_L > 0 and T_eff < min_L:
                    continue
                if min_nnz > 0 and (nnz.get(u, 0) < min_nnz or nnz.get(v, 0) < min_nnz):
                    continue
                if not cond_q and not cond_r:
                    continue
                if stab_mat is not None and not stab_mat.empty:
                    s = float(stab_mat.at[u, v]) if (u in stab_mat.index and v in stab_mat.columns) else 0.0
                    if s < stability_min:
                        continue
                # Safe numeric serialization (no NaN in JSON)
                def _safe(df: pd.DataFrame, key_u: str, key_v: str) -> float | None:
                    try:
                        if df.empty or key_u not in df.index or key_v not in df.columns:
                            return None
                        val = float(df.at[key_u, key_v])
                        return None if pd.isna(val) else val
                    except Exception:
                        return None
                jv = _safe(jacc, u, v)
                rv_corr = _safe(corr, u, v)
                # coincidences
                try:
                    coincidences = int((binf[u] & binf[v]).sum()) if (binf is not None and not binf.empty) else None
                except Exception:
                    coincidences = None
                rows.append({
                    "u": u,
                    "v": v,
                    "score": (None if pd.isna(xr) else float(xr)),
                    "r": (None if pd.isna(xr) else float(xr)),
                    "lag_seconds": (None if pd.isna(lag_s) else float(lag_s)),
                    "p": None,
                    "q": (None if pd.isna(qv) else float(qv)),
                    "jaccard": jv,
                    "corr": rv_corr,
                    "source": "xcorr",
                    "L": T_eff or None,
                    "nnz_u": nnz.get(u),
                    "nnz_v": nnz.get(v),
                    "coincidences": coincidences,
                    "stability": (float(stab_mat.at[u, v]) if (stab_mat is not None and not stab_mat.empty and u in stab_mat.index and v in stab_mat.columns) else None),
                })
        # Fallback if empty
        if not rows:
            # use correlation/jaccard thresholds as fallback
            try:
                corr_t = float(request.args.get("corr_threshold", "0.6"))
            except Exception:
                corr_t = 0.6
            try:
                jacc_t = float(request.args.get("jacc_threshold", "0.18"))
            except Exception:
                jacc_t = 0.18
            used = set()
            if not corr.empty:
                cols = corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        u, v = cols[i], cols[j]
                        rv = float(corr.iat[i, j])
                        if abs(rv) >= corr_t:
                            used.add((u, v))
                            rows.append({"u": u, "v": v, "corr": rv, "score": rv, "source": "corr"})
            if not jacc.empty:
                cols = jacc.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        u, v = cols[i], cols[j]
                        if (u, v) in used:
                            continue
                        jv = float(jacc.iat[i, j])
                        if jv >= jacc_t:
                            rows.append({"u": u, "v": v, "jaccard": jv, "score": jv, "source": "jaccard"})
    elif method in ("spearman", "pearson"):
        _c, qpmat, qqmat = analyzer.compute_correlation_pq(method=method, residualize=residualize, fdr_method=fdr_method)
        mat = _c
        qmat = qqmat
        users = list(mat.columns)
        # sample stats and optional stability
        frame = analyzer.df_resid if (residualize and getattr(analyzer, 'df_resid', None) is not None) else analyzer.df_for_corr
        T_eff = len(frame.index) if frame is not None and not frame.empty else 0
        binf = analyzer._binary_frame()
        nnz = {u: int(binf[u].sum()) for u in users} if (binf is not None and not binf.empty) else {u: 0 for u in users}
        stab_mat = None
        if stable:
            stab_mat = analyzer.stability_matrix(method=stability_method, residualize=residualize, folds=folds, min_abs=min_abs)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u, v = users[i], users[j]
                r = float(mat.iat[i, j])
                qv = float(qmat.iat[i, j]) if not qmat.empty else float('nan')
                if min_L > 0 and T_eff < min_L:
                    continue
                if min_nnz > 0 and (nnz.get(u, 0) < min_nnz or nnz.get(v, 0) < min_nnz):
                    continue
                cond_r = abs(r) >= min_abs
                cond_q = (qthr_val is not None and not pd.isna(qv) and qv <= qthr_val)
                if not cond_r and not cond_q:
                    continue
                if stab_mat is not None and not stab_mat.empty:
                    s = float(stab_mat.at[u, v]) if (u in stab_mat.index and v in stab_mat.columns) else 0.0
                    if s < stability_min:
                        continue
                try:
                    coincidences = int((binf[u] & binf[v]).sum()) if (binf is not None and not binf.empty) else None
                except Exception:
                    coincidences = None
                rows.append({"u": u, "v": v, "score": r, "r": r, "q": (None if pd.isna(qv) else float(qv)), "source": method, "L": T_eff or None, "nnz_u": nnz.get(u), "nnz_v": nnz.get(v), "coincidences": coincidences, "stability": (float(stab_mat.at[u, v]) if (stab_mat is not None and not stab_mat.empty and u in stab_mat.index and v in stab_mat.columns) else None)})
        rows = sorted(rows, key=lambda x: ((x.get("q", 1.0) if x.get("q") is not None else 1.0), -abs(x.get("score", 0.0))))[:top]
    elif method == "te":
        # Transfer Entropy with bootstrap null and BY/BH FDR
        te, pmat, qmat = analyzer.compute_transfer_entropy(
            residualize=residualize,
            quantize=te_quantize,
            perms=perms,
            bootstrap=te_bootstrap,
            block_p=te_block_p,
            fdr_method=fdr_method,
        )
        users = list(te.columns)
        for i in range(len(users)):
            for j in range(len(users)):
                if i == j:
                    continue
                u, v = users[i], users[j]
                try:
                    te_val = float(te.at[u, v]) if not te.empty else float("nan")
                except Exception:
                    te_val = float("nan")
                try:
                    qv = float(qmat.at[u, v]) if (not qmat.empty) else float("nan")
                except Exception:
                    qv = float("nan")
                cond_q = (qthr_val is not None) and (not pd.isna(qv)) and (qv <= qthr_val)
                cond_s = (not pd.isna(te_val)) and (abs(te_val) >= min_abs)
                if not cond_q and not cond_s:
                    continue
                rows.append({
                    "u": u,
                    "v": v,
                    "score": (None if pd.isna(te_val) else float(te_val)),
                    "q": (None if pd.isna(qv) else float(qv)),
                    "source": "te"
                })
        rows = sorted(rows, key=lambda x: (x.get("q", 1.0) if x.get("q") is not None else 1.0, -(x.get("score", 0.0) or 0.0)))[:top]
    elif method == "hawkes":
        # Directed excitation with circular-shift null
        hl = float(request.args.get("half_life_minutes", "10"))
        maxlag = request.args.get("max_lag_minutes")
        maxlag_f = float(maxlag) if maxlag is not None else None
        S, P, Q = analyzer.hawkes_excitation(half_life_minutes=hl, max_lag_minutes=maxlag_f, perms=perms, fdr_method=fdr_method)
        users = list(S.columns)
        for i in range(len(users)):
            for j in range(len(users)):
                if i == j:
                    continue
                u, v = users[i], users[j]
                s = float(S.at[u, v]) if (not S.empty) else float('nan')
                qv = float(Q.at[u, v]) if (not Q.empty) else float('nan')
                cond_q = (qthr_val is not None) and (not pd.isna(qv)) and (qv <= qthr_val)
                cond_s = (not pd.isna(s)) and (s >= min_abs)
                if not cond_q and not cond_s:
                    continue
                rows.append({"u": u, "v": v, "score": s, "q": (None if pd.isna(qv) else float(qv)), "source": "hawkes"})
        rows = sorted(rows, key=lambda x: (x.get("q", 1.0) if x.get("q") is not None else 1.0, -(x.get("score", 0.0) or 0.0)))[:top]
    elif method in ("mcc", "kappa", "ochiai", "overlap"):
        if method == "mcc":
            mat = analyzer.mcc_matrix()
        elif method == "kappa":
            mat = analyzer.kappa_matrix()
        elif method == "ochiai":
            mat = analyzer.ochiai_matrix()
        else:
            mat = analyzer.overlap_matrix()
        users = list(mat.columns)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u, v = users[i], users[j]
                val = float(mat.iat[i, j])
                rows.append({"u": u, "v": v, "score": val, "source": method})
        rows = sorted(rows, key=lambda x: abs(x.get("score", 0.0)), reverse=True)[:top]
    elif method == "jaccard":
        jac = analyzer.get_jaccard_matrix()
        qmat = None
        if permute:
            _p, qmat = analyzer.jaccard_perm_pvals(perms=perms, fdr_method=fdr_method)
        users = list(jac.columns)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u, v = users[i], users[j]
                jv = float(jac.iat[i, j])
                qv = float(qmat.iat[i, j]) if (qmat is not None and not qmat.empty) else float('nan')
                cond = (jv >= float(request.args.get("jacc_threshold", "0.18"))) or (qthr_val is not None and not pd.isna(qv) and qv <= qthr_val)
                if not cond:
                    continue
                rows.append({"u": u, "v": v, "score": jv, "jaccard": jv, "q": (None if pd.isna(qv) else float(qv)), "source": "jaccard"})
        rows = sorted(rows, key=lambda x: ((x.get("q", 1.0) if x.get("q") is not None else 1.0), -(x.get("score", 0.0) or 0.0)))[:top]
    elif method == "esi":
        esi = analyzer.event_sync_index(tau_seconds=tau_seconds)
        users = list(esi.columns)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u, v = users[i], users[j]
                val = float(esi.iat[i, j])
                rows.append({"u": u, "v": v, "score": val, "source": "esi"})
        rows = sorted(rows, key=lambda x: (x.get("score", 0.0)), reverse=True)[:top]
    else:
        rows = []
    
    result = {"pairs": rows, "meta": meta}
    _cache_set(cache_key, result)  # Cache for 5 minutes
    return jsonify(result)


@api_bp.get("/graph/combined")
def graph_combined() -> Any:
    analyzer, _ = _build_analyzer_from_request()
    corr_t = float(request.args.get("corr_threshold", str(analyzer.default_correlation_threshold)))
    jacc_t = float(request.args.get("jacc_threshold", str(analyzer.default_jaccard_threshold)))
    q_t = float(request.args.get("q_threshold", str(analyzer.fdr_alpha)))
    method = (request.args.get("method") or "xcorr").lower()
    residualize = (request.args.get("residualize") or "false").lower() == "true"
    fdr_method = (request.args.get("fdr_method") or "by").lower()
    layout = request.args.get("layout", "circle")  # circle|spring
    # spring params
    k = request.args.get("k")
    iterations = int(request.args.get("iterations", "50"))
    # For now, use combined graph logic with corr/xcorr/jaccard; when method is
    # not xcorr, we still include corr/jaccard edges. Residualization influences
    # corr via compute_correlation_matrix.
    g = analyzer.build_combined_graph(corr_threshold=corr_t, jaccard_threshold=jacc_t, qval_threshold=q_t, residualize=residualize, fdr_method=fdr_method, use_runtime_xcorr=True)
    # Compute positions
    positions: Dict[str, Dict[str, float]] = {}
    try:
        import networkx as nx  # type: ignore
        if layout == "spring" and g.number_of_nodes() > 0:
            pos = nx.spring_layout(g, k=float(k) if k is not None else None, iterations=iterations, seed=42)
        else:
            # default: circle
            pos = nx.circular_layout(g)
        positions = {n: {"x": float(p[0]), "y": float(p[1])} for n, p in pos.items()}
    except Exception as e:
        tui.tui_print_warning(f"layout compute error: {e}")

    nodes = [{"id": n, "community": g.nodes[n].get("community"), **(positions.get(n,{"x":0.0,"y":0.0}))} for n in g.nodes]
    edges = [{
        "source": u,
        "target": v,
        "weight": d.get("weight"),
        "corr": d.get("corr"),
        "crosscorr": d.get("crosscorr"),
        "qval": d.get("qval"),
        "jaccard": d.get("jaccard"),
    } for u, v, d in g.edges(data=True)]
    return jsonify({"nodes": nodes, "edges": edges, "meta": {"method": method, "residualize": residualize, "since": request.args.get("since")}})


@api_bp.get("/consensus/top")
def consensus_top() -> Any:
    """Aggregate multiple metrics to produce top N \"consensus\" pairs with explanations.

    Query:
      - since, residualize=bool, permute=bool, perms=int, tau_seconds=int, limit=int
    Returns:
      { pairs: [{u,v, score, agree_count, reason, leader?: string, lag_seconds?: number, metrics: {...}}], meta: {...} }
    """
    # Use full fingerprint for comprehensive caching
    fingerprint = _get_request_fingerprint()
    permute = (request.args.get("permute") or "false").lower() == "true"
    perms = max(50, min(200, int(request.args.get("perms", "100"))))
    ck = f"consensus:{permute}:{perms}:{fingerprint}"
    cached = _cache_get(ck)
    if cached is not None:
        return jsonify(cached)
    analyzer, _ = _build_analyzer_from_request()
    residualize = (request.args.get("residualize") or "false").lower() == "true"
    permute = (request.args.get("permute") or "false").lower() == "true"
    perms = max(50, min(200, int(request.args.get("perms", "100"))))
    jobs = max(0, int(request.args.get("jobs", "0")))
    bsz = max(1, min(64, int(request.args.get("batch_size", "16"))))
    tau_seconds = int(request.args.get("tau_seconds", "120"))
    limit = max(1, min(10, int(request.args.get("limit", "3"))))
    qthr = float(request.args.get("q_threshold", "0.05"))
    min_abs = float(request.args.get("min_abs_corr", "0.3"))
    fdr_method = (request.args.get("fdr_method") or "by").lower()
    use_te = (request.args.get("use_te") or "true").lower() == "true"
    permute_jaccard = (request.args.get("permute_jaccard") or "true").lower() == "true"
    te_bootstrap = (request.args.get("bootstrap") or "stationary").lower()
    te_block_p = float(request.args.get("block_p", "0.1"))

    # Prepare frames/matrices
    if residualize:
        try:
            analyzer.residualize_time()
        except Exception:
            pass
    corr = analyzer.compute_correlation_matrix(method="spearman", residualize=residualize)
    xcorr, lag, _pv, qvals = analyzer.compute_crosscorr_with_lag(method=analyzer.corr_method, residualize=residualize, fdr_method=fdr_method)
    if permute:
        kw = {"perms": perms, "method": analyzer.corr_method, "use_residuals": residualize, "batch_size": bsz, "fdr_method": fdr_method}
        if jobs > 0:
            kw["n_jobs"] = jobs
        _p, qvals = analyzer.crosscorr_pvals_max_lag_perm(**kw)
    jacc = analyzer.get_jaccard_matrix()
    jacc_q = None
    if permute_jaccard:
        _p_j, jacc_q = analyzer.jaccard_perm_pvals(perms=perms, fdr_method=fdr_method)
    mcc = analyzer.mcc_matrix()
    kappa = analyzer.kappa_matrix()
    och = analyzer.ochiai_matrix()
    ovl = analyzer.overlap_matrix()
    esi = analyzer.event_sync_index(tau_seconds=tau_seconds)
    te = None
    te_q = None
    if use_te:
        te, _p_te, te_q = analyzer.compute_transfer_entropy(residualize=residualize, perms=perms, bootstrap=te_bootstrap, block_p=te_block_p, fdr_method=fdr_method)
    # co-changepoints
    cp_use = (request.args.get("use_cp") or "true").lower() == "true"
    cp_tau_steps = int(request.args.get("cp_tau_steps", "3"))
    cp = None
    cp_q = None
    if cp_use:
        try:
            cp, _cp_p, cp_q = analyzer.co_changepoints_matrix(residualize=residualize, tau_steps=cp_tau_steps, perms=perms, fdr_method=fdr_method)
        except Exception:
            cp, cp_q = None, None

    users = list(corr.columns) if not corr.empty else analyzer.user_list
    # Default gates: require reasonable data volume
    min_L = int(request.args.get("min_L", "100"))
    min_nnz = int(request.args.get("min_nonzero", "10"))
    frame = analyzer.df_resid if (residualize and getattr(analyzer, 'df_resid', None) is not None) else analyzer.df_for_corr
    T_eff = len(frame.index) if frame is not None and not frame.empty else 0
    rows: List[Dict[str, Any]] = []
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            u, v = users[i], users[j]
            try:
                r = float(corr.at[u, v]) if not corr.empty else float("nan")
            except Exception:
                r = float("nan")
            try:
                xr = float(xcorr.at[u, v]) if not xcorr.empty else float("nan")
            except Exception:
                xr = float("nan")
            try:
                q = float(qvals.at[u, v]) if not qvals.empty else float("nan")
            except Exception:
                q = float("nan")
            try:
                lag_s = float(lag.at[u, v]) if not lag.empty else float("nan")
            except Exception:
                lag_s = float("nan")
            jj = float(jacc.at[u, v]) if (not jacc.empty and u in jacc.index and v in jacc.columns) else float("nan")
            jq = float(jacc_q.at[u, v]) if (jacc_q is not None and not jacc_q.empty and u in jacc_q.index and v in jacc_q.columns) else float("nan")
            mm = float(mcc.at[u, v]) if (not mcc.empty and u in mcc.index and v in mcc.columns) else float("nan")
            kk = float(kappa.at[u, v]) if (not kappa.empty and u in kappa.index and v in kappa.columns) else float("nan")
            oo = float(och.at[u, v]) if (not och.empty and u in och.index and v in och.columns) else float("nan")
            ov = float(ovl.at[u, v]) if (not ovl.empty and u in ovl.index and v in ovl.columns) else float("nan")
            ee = float(esi.at[u, v]) if (not esi.empty and u in esi.index and v in esi.columns) else float("nan")
            tedir = float(te.at[u, v]) if (te is not None and not te.empty and u in te.index and v in te.columns) else float("nan")
            teq = float(te_q.at[u, v]) if (te_q is not None and not te_q.empty and u in te_q.index and v in te_q.columns) else float("nan")
            ccp = float(cp.at[u, v]) if (cp_use and cp is not None and not cp.empty and u in cp.index and v in cp.columns) else float('nan')
            ccpq = float(cp_q.at[u, v]) if (cp_use and cp_q is not None and not cp_q.empty and u in cp_q.index and v in cp_q.columns) else float('nan')

            # Threshold checks
            agree_flags = {
                "q": (not pd.isna(q)) and q <= qthr,
                "xcorr": (not pd.isna(xr)) and abs(xr) >= min_abs,
                "spearman": (not pd.isna(r)) and abs(r) >= min_abs,
                "jaccard": (not pd.isna(jj)) and jj >= 0.18 or ((jacc_q is not None) and (not pd.isna(jq)) and jq <= qthr),
                "mcc": (not pd.isna(mm)) and mm >= 0.2,
                "kappa": (not pd.isna(kk)) and kk >= 0.2,
                "ochiai": (not pd.isna(oo)) and oo >= 0.4,
                "overlap": (not pd.isna(ov)) and ov >= 0.4,
                "esi": (not pd.isna(ee)) and ee >= 0.3,
                "te": (use_te and (not pd.isna(teq)) and teq <= qthr) or (use_te and (not pd.isna(tedir)) and abs(tedir) >= min_abs),
                "cp": (cp_use and ((not pd.isna(ccpq)) and ccpq <= qthr)) or (cp_use and (not pd.isna(ccp)) and ccp >= 0.5),
            }
            # Gates by size and support
            if T_eff < min_L:
                continue
            # Roughly estimate support from binary matrix
            binf = analyzer._binary_frame()
            nnz_u = int(binf[u].sum()) if (binf is not None and not binf.empty and u in binf.columns) else 0
            nnz_v = int(binf[v].sum()) if (binf is not None and not binf.empty and v in binf.columns) else 0
            if nnz_u < min_nnz or nnz_v < min_nnz:
                continue

            agree_count = int(sum(1 for vflag in agree_flags.values() if vflag))
            if agree_count == 0:
                continue
            # Composite score (bounded ~[0,9])
            score = (
                (0 if pd.isna(q) else (1.0 - min(max(q, 0.0), 1.0))) +
                (0 if pd.isna(xr) else abs(xr)) +
                (0 if pd.isna(r) else abs(r)) +
                (0 if pd.isna(jj) else jj) +
                (0 if pd.isna(mm) else mm) +
                (0 if pd.isna(kk) else kk) +
                (0 if pd.isna(oo) else oo) +
                (0 if pd.isna(ov) else ov) +
                (0 if pd.isna(ee) else ee)
            )
            # Leader inference from lag: positive -> v lags behind u, so u leads
            leader = None
            if not pd.isna(lag_s):
                leader = u if lag_s > 0 else (v if lag_s < 0 else None)
            reason = (
                f"q={q if not pd.isna(q) else None}, |xcorr|={abs(xr) if not pd.isna(xr) else None}, |Spearman|={abs(r) if not pd.isna(r) else None}, "
                f"Jaccard={jj if not pd.isna(jj) else None}, jq={jq if not pd.isna(jq) else None}, MCC={mm if not pd.isna(mm) else None}, κ={kk if not pd.isna(kk) else None}, "
                f"Ochiai={oo if not pd.isna(oo) else None}, Overlap={ov if not pd.isna(ov) else None}, ESI={ee if not pd.isna(ee) else None}, TE={tedir if not pd.isna(tedir) else None}, TEq={teq if not pd.isna(teq) else None}, CP={ccp if not pd.isna(ccp) else None}, CPq={ccpq if not pd.isna(ccpq) else None}"
            )
            rows.append({
                "u": u, "v": v, "score": float(score), "agree_count": agree_count,
                "leader": leader, "lag_seconds": (None if pd.isna(lag_s) else float(lag_s)),
                "metrics": {"q": None if pd.isna(q) else float(q), "xcorr": None if pd.isna(xr) else float(xr),
                             "spearman": None if pd.isna(r) else float(r), "jaccard": None if pd.isna(jj) else float(jj), "jaccard_q": None if pd.isna(jq) else float(jq),
                             "mcc": None if pd.isna(mm) else float(mm), "kappa": None if pd.isna(kk) else float(kk),
                             "ochiai": None if pd.isna(oo) else float(oo), "overlap": None if pd.isna(ov) else float(ov),
                             "esi": None if pd.isna(ee) else float(ee), "te": None if pd.isna(tedir) else float(tedir), "te_q": None if pd.isna(teq) else float(teq),
                             "cp": None if pd.isna(ccp) else float(ccp), "cp_q": None if pd.isna(ccpq) else float(ccpq)},
                "reason": reason,
            })
    # Sort by agree_count then score desc
    rows.sort(key=lambda r: (r.get("agree_count", 0), r.get("score", 0.0)), reverse=True)
    # NBS component significance by co-CP
    nbs_p = None
    try:
        if cp_use and cp is not None and cp_q is not None:
            nbs_p = analyzer.nbs_component_pvalue_from_cp(cp, cp_q, tau_steps=cp_tau_steps, alpha=qthr, perms=perms)
    except Exception:
        nbs_p = None
    # Estimate E[FP] 
    efp_upper = None
    try:
        R = sum(1 for r in rows if (r.get('metrics',{}).get('q') is not None and r['metrics']['q'] <= qthr))
        efp_upper = float(qthr) * float(R)
    except Exception:
        efp_upper = None
    out = {"pairs": rows[:limit], "meta": {"residualize": residualize, "permute": permute, "perms": perms, "jobs": (jobs or None), "tau_seconds": tau_seconds, "since": request.args.get("since"), "nbs_p": nbs_p, "efp_upper": efp_upper}}
    _cache_set(ck, out)
    return jsonify(out)


# ---- Process control: start/stop/status/logs for tracker ----

def _project_root() -> str:
    # api.py is tgTrax/web/api.py → project root is tgTrax/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pid_file() -> str:
    return os.path.join(_project_root(), ".tracker.api.pid")


def _log_file() -> str:
    return os.path.join(_project_root(), "tracker.api.log")


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@api_bp.get("/stack/status")
def stack_status() -> Any:
    pid = None
    running = False
    try:
        if os.path.exists(_pid_file()):
            with open(_pid_file(), "r") as f:
                pid = int(f.read().strip())
            running = _is_running(pid)
            if not running:
                # stale
                os.remove(_pid_file())
                pid = None
    except Exception as e:
        tui.tui_print_warning(f"status check error: {e}")
    # include started_at if exists
    started_at = None
    try:
        import json
        with open(_state_file(), 'r') as f:
            st = json.load(f)
            started_at = st.get('started_at')
    except Exception:
        pass
    return jsonify({"running": running, "pid": pid, "started_at": started_at})


# ---- Authorization helpers ----
AUTH_STATE: Dict[str, Any] = {"loop": None, "client": None, "thread": None}


def _get_api_credentials() -> Tuple[int | None, str | None, str]:
    api_id_str = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session_name = os.getenv("TELEGRAM_SESSION_NAME", "tgTrax_session")
    api_id = None
    try:
        if api_id_str:
            api_id = int(api_id_str)
    except ValueError:
        api_id = None
    # Normalize session to absolute path under tgTrax/sessions
    sessions_dir = os.path.join(_project_root(), "sessions")
    try:
        os.makedirs(sessions_dir, exist_ok=True)
    except Exception:
        pass
    session_path = os.path.join(sessions_dir, session_name)
    # Migrate legacy session from repo root if present
    legacy = os.path.join(os.path.dirname(_project_root()), f"{session_name}.session")
    try:
        if os.path.exists(legacy) and not os.path.exists(f"{session_path}.session"):
            import shutil
            shutil.copy2(legacy, f"{session_path}.session")
            tui.tui_print_info(f"Migrated legacy session to {session_path}.session")
    except Exception as e:
        tui.tui_print_warning(f"Session migration skipped: {e}")
    return api_id, api_hash, session_path


def _ensure_auth_loop() -> None:
    if AUTH_STATE.get("thread") and AUTH_STATE.get("loop"):
        return
    ready = threading.Event()

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        AUTH_STATE["loop"] = loop
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=runner, name="tgtrax-auth-loop", daemon=True)
    AUTH_STATE["thread"] = t
    t.start()
    ready.wait(timeout=3.0)


def _auth_run(coro):
    loop = AUTH_STATE.get("loop")
    if not loop:
        _ensure_auth_loop()
        loop = AUTH_STATE.get("loop")
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=30)


async def _auth_connect_and_send_code(phone: str) -> Dict[str, Any]:
    api_id, api_hash, session = _get_api_credentials()
    if not api_id or not api_hash:
        return {"ok": False, "error": "Missing TELEGRAM_API_ID or TELEGRAM_API_HASH"}
    client = AUTH_STATE.get("client")
    if client is None:
        client = TelegramClient(session, api_id, api_hash)
        AUTH_STATE["client"] = client
    await client.connect()
    sc = await client.send_code_request(phone)
    return {"ok": True, "type": sc.__class__.__name__}


async def _auth_sign_in(phone: str, code: str, password: str | None = None) -> Dict[str, Any]:
    api_id, api_hash, session = _get_api_credentials()
    if not api_id or not api_hash:
        return {"ok": False, "error": "Missing TELEGRAM_API_ID or TELEGRAM_API_HASH"}
    client: TelegramClient | None = AUTH_STATE.get("client")
    if client is None:
        client = TelegramClient(session, api_id, api_hash)
        AUTH_STATE["client"] = client
    await client.connect()
    try:
        await client.sign_in(phone=phone, code=code)
    except tg_errors.SessionPasswordNeededError:
        if not password:
            return {"ok": False, "need_2fa": True}
        await client.sign_in(password=password)
    ok = await client.is_user_authorized()
    await client.disconnect()
    AUTH_STATE["client"] = None
    return {"ok": ok}


def _auth_is_authorized_light() -> Dict[str, Any]:
    """Lightweight check that does NOT open Telethon client to avoid session locks.

    Reads the SQLite session file in read-only mode and checks for non-empty auth_key.
    """
    import sqlite3
    _api_id, _api_hash, session_path = _get_api_credentials()
    sess_file = f"{session_path}.session"
    if not os.path.exists(sess_file):
        return {"ok": True, "authorized": False, "reason": "no_session_file"}
    try:
        conn = sqlite3.connect(f"file:{sess_file}?mode=ro", uri=True, timeout=0.2)
        try:
            cur = conn.cursor()
            cur.execute("SELECT auth_key FROM sessions LIMIT 1")
            row = cur.fetchone()
            if not row:
                return {"ok": True, "authorized": False, "reason": "no_rows"}
            auth_key = row[0]
            return {"ok": True, "authorized": bool(auth_key)}
        finally:
            conn.close()
    except Exception as e:
        return {"ok": False, "authorized": False, "error": str(e)}


@api_bp.get("/auth/status")
def auth_status() -> Any:
    res = _auth_is_authorized_light()
    return jsonify(res)


@api_bp.post("/auth/send_code")
def auth_send_code() -> Any:
    body = request.get_json(silent=True) or {}
    phone = body.get("phone") or os.getenv("TELEGRAM_PHONE_NUMBER")
    if not phone:
        return jsonify({"ok": False, "error": "Phone not provided and TELEGRAM_PHONE_NUMBER missing"}), 400
    try:
        _ensure_auth_loop()
        res = _auth_run(_auth_connect_and_send_code(phone))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.post("/auth/verify")
def auth_verify() -> Any:
    body = request.get_json(silent=True) or {}
    phone = body.get("phone") or os.getenv("TELEGRAM_PHONE_NUMBER")
    code = body.get("code")
    password = body.get("password")
    if not phone or not code:
        return jsonify({"ok": False, "error": "Missing phone or code"}), 400
    try:
        _ensure_auth_loop()
        res = _auth_run(_auth_sign_in(phone, code, password))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.post("/stack/start")
def stack_start() -> Any:
    # Start tracker in a subprocess: python -m tgTrax.main tracker
    # If already running, no-op
    try:
        if os.path.exists(_pid_file()):
            with open(_pid_file(), "r") as f:
                pid = int(f.read().strip())
            if _is_running(pid):
                return jsonify({"started": False, "message": "Tracker already running", "pid": pid}), 200
            else:
                os.remove(_pid_file())
    except Exception:
        pass

    logf = _log_file()
    pidf = _pid_file()
    try:
        env = os.environ.copy()
        # Ensure PYTHONPATH includes project root
        env["PYTHONPATH"] = f"{_project_root()}:{env.get('PYTHONPATH','')}"
        # Start process in project root
        # Truncate log to make fresh run clear
        with open(logf, "wb", buffering=0) as lf:
            lf.write(b"--- tracker start ---\n")
        with open(logf, "ab", buffering=0) as lf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "tgTrax.main", "tracker"],
                cwd=_project_root(),
                stdout=lf,
                stderr=lf,
                env=env,
            )
        with open(pidf, "w") as pf:
            pf.write(str(proc.pid))
        # write state
        try:
            import json, datetime as dt
            with open(_state_file(), 'w') as f:
                json.dump({"pid": proc.pid, "started_at": dt.datetime.utcnow().isoformat() + 'Z'}, f)
        except Exception as e:
            tui.tui_print_warning(f"state write failed: {e}")
        time.sleep(0.5)
        return jsonify({"started": True, "pid": proc.pid})
    except Exception as e:
        return jsonify({"started": False, "error": str(e)}), 500


@api_bp.post("/stack/stop")
def stack_stop() -> Any:
    if not os.path.exists(_pid_file()):
        return jsonify({"stopped": False, "message": "No PID file"}), 404
    try:
        with open(_pid_file(), "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        if os.path.exists(_pid_file()):
            os.remove(_pid_file())
        try:
            import json, datetime as dt
            with open(_state_file(), 'w') as f:
                json.dump({"stopped_at": dt.datetime.utcnow().isoformat() + 'Z'}, f)
        except Exception:
            pass
        return jsonify({"stopped": True})
    except Exception as e:
        return jsonify({"stopped": False, "error": str(e)}), 500


@api_bp.get("/stack/logs")
def stack_logs() -> Any:
    lines = int(request.args.get("lines", "100"))
    p = _log_file()
    if not os.path.exists(p):
        return jsonify({"log": "(no log yet)"})
    try:
        # Read last N lines efficiently
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= lines:
                step = block if size - block > 0 else size
                size -= step
                f.seek(size)
                data = f.read(step) + data
            text = data.decode(errors="replace")
            tail = "\n".join(text.splitlines()[-lines:])
        return jsonify({"log": tail})
    except Exception as e:
        return jsonify({"log": f"(error reading log: {e})"})
