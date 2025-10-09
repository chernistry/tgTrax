from __future__ import annotations

# ==== TEMPORAL ANALYSIS MODULE ==== #
"""
Temporal analysis utilities for user activity data.

Provides the `TemporalAnalyzer` for correlation, cross-correlation with lag
search, community detection, and summary computations, keeping original logic
intact. Includes helper routines for residualization and Gantt preparation.
"""


import datetime  # Added for type hinting in activity interval utilities
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Optional  # Precise type hints

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr  # For p-values and cross-correlation

try:
    from scipy.stats import t as student_t  # For fast p-values via t distribution
except Exception:
    student_t = None  # type: ignore
try:
    from scipy.stats import beta as beta_dist  # For Clopper–Pearson intervals
except Exception:
    beta_dist = None  # type: ignore
try:
    from scipy.stats import ConstantInputWarning  # type: ignore
except Exception:

    class ConstantInputWarning(Warning):
        pass


import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import threading


# ==== CORE PROCESSING MODULE ==== #
# --► DATA EXTRACTION & TRANSFORMATION


def _xcorr_perm_worker_batch(
    arr: np.ndarray,
    T: int,
    n: int,
    max_lag_steps: int,
    obs_max: np.ndarray,
    batch_shifts: np.ndarray,
) -> np.ndarray:
    """
    Process worker: compute greater-equal counts for a batch of circular shifts.

    Args:
        arr: Ranked (Spearman) or raw (Pearson) T×n matrix.
        T: Number of timesteps.
        n: Number of users (columns).
        max_lag_steps: Max lag in steps for search window.
        obs_max: Observed max |r| per pair matrix.
        batch_shifts: Vector of circular shifts to test.

    Returns:
        np.ndarray: Symmetric GE-counts increment (n×n) for this batch.
    """
    iu_local = np.triu_indices(n, k=1)
    ge = np.zeros((n, n), dtype=float)
    for s in batch_shifts:
        Yshift = np.roll(arr, int(s), axis=0)
        best_abs = np.full((n, n), -1.0, dtype=float)
        for k in range(-max_lag_steps, max_lag_steps + 1):
            # k is the lag in steps. We extract overlapping windows X and Y.
            # L is the overlap length between these windows.
            if k == 0:
                L = T
                X = arr
                Y = Yshift
            elif k > 0:
                L = T - k
                if L < 3:
                    continue
                X = arr[k:, :]
                Y = Yshift[:L, :]
            else:
                kk = -k
                L = T - kk
                if L < 3:
                    continue
                X = arr[:L, :]
                Y = Yshift[kk:, :]
            # Column/Pairwise compact stats for fast correlation across all pairs:
            # - sx/sy: column sums; ssx/ssy: sum of squares; sxy: X'Y cross-products
            # - num: covariance numerators; vx/vy: variance parts per column
            sx = X.sum(axis=0)
            sy = Y.sum(axis=0)
            ssx = np.einsum("ij,ij->j", X, X)
            ssy = np.einsum("ij,ij->j", Y, Y)
            sxy = X.T @ Y
            num = sxy - np.outer(sx, sy) / L
            vx = ssx - (sx * sx) / L
            vy = ssy - (sy * sy) / L
            with np.errstate(invalid="ignore", divide="ignore"):
                R = num / np.sqrt(np.outer(vx, vy))
            R_abs = np.abs(R)
            mask = R_abs[iu_local] > best_abs[iu_local]
            if np.any(mask):
                best_abs[iu_local] = np.where(mask, R_abs[iu_local], best_abs[iu_local])
        ge_mask = best_abs[iu_local] >= obs_max[iu_local]
        if np.any(ge_mask):
            ii = iu_local[0][ge_mask]
            jj = iu_local[1][ge_mask]
            ge[ii, jj] += 1.0
            ge[jj, ii] += 1.0
    return ge


try:
    from statsmodels.tsa.seasonal import STL  # Optional, for seasonal adjustment

    _HAS_STL = True
except Exception:
    _HAS_STL = False

try:
    from scipy.signal import fftconvolve  # type: ignore

    _HAS_FFTCONV = True
except Exception:
    _HAS_FFTCONV = False

try:
    import numba  # type: ignore
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# ---- Shared memory helpers for permutation workers ----
_SHM_ARR_NAME: str | None = None
_SHM_ARR_SHAPE: Tuple[int, int] | None = None
_SHM_ARR_DTYPE: str | None = None
_SHM_ARR_CACHE = threading.local()


def _init_worker_shm(name: str, shape: Tuple[int, int], dtype_str: str) -> None:
    global _SHM_ARR_NAME, _SHM_ARR_SHAPE, _SHM_ARR_DTYPE
    _SHM_ARR_NAME = name
    _SHM_ARR_SHAPE = shape
    _SHM_ARR_DTYPE = dtype_str
    _SHM_ARR_CACHE.arr = None


def _get_shm_array() -> np.ndarray:
    if getattr(_SHM_ARR_CACHE, "arr", None) is not None:
        return _SHM_ARR_CACHE.arr  # type: ignore
    assert _SHM_ARR_NAME and _SHM_ARR_SHAPE and _SHM_ARR_DTYPE
    shm = shared_memory.SharedMemory(name=_SHM_ARR_NAME)
    arr = np.ndarray(_SHM_ARR_SHAPE, dtype=np.dtype(_SHM_ARR_DTYPE), buffer=shm.buf)
    _SHM_ARR_CACHE.arr = arr
    return arr


def _xcorr_perm_worker_batch_shm(
    T: int,
    n: int,
    max_lag_steps: int,
    obs_max: np.ndarray,
    batch_shifts: np.ndarray,
) -> np.ndarray:
    arr = _get_shm_array()
    return _xcorr_perm_worker_batch(arr, T, n, max_lag_steps, obs_max, batch_shifts)


import community.community_louvain as community_louvain  # Explicitly for clarity
import logging
from tgTrax.utils.cache import cache_get, cache_set, khash, default_ttl

logger = logging.getLogger(__name__)


# --- Constants ---
# (No module-level constants defined after removals)


# --- TemporalAnalyzer Class ---
class TemporalAnalyzer:
    """
    Analyze temporal patterns and relationships across user activity series.

    Focus areas:
    - Correlation and cross-correlation (with lag search and FDR correction)
    - Binary-activity Jaccard similarity
    - Optional residualization by time-of-week and global factor
    - Lightweight community detection support
    """

    def __init__(
        self,
        activity_df: pd.DataFrame,
        resample_period: str = "1min",
        correlation_threshold: float = 0.6,
        jaccard_threshold: float = 0.18,
        debug_users: Optional[List[str]] = None,
        # New preprocessing options
        ewma_alpha: Optional[float] = None,
        seasonal_adjust: bool = False,
        # Resampling/NaN policy
        resample_agg: str = "max",  # one of: max|mean|sum|any
        fill_missing: str = "zero",  # one of: zero|ffill|nan
        # Cross-correlation options
        max_lag_minutes: int = 15,
        corr_method: str = "spearman",
        # Multiple testing correction
        fdr_alpha: float = 0.05,
        # Residualization defaults
        residual_include_global: bool = False,
        residual_prior_strength: int = 24,
        residual_variance_stabilize: bool = True,
    ):
        """
        Initialize analyzer with activity data and preprocessing options.

        Args:
            activity_df: DateTimeIndex DataFrame; columns are users; values 0/1.
            resample_period: Resampling rule or 'auto'.
            correlation_threshold: Default correlation significance threshold.
            jaccard_threshold: Default Jaccard significance threshold.
            debug_users: Optional list of usernames for compact debug samples.
            ewma_alpha: Optional EWMA smoothing factor in (0, 1].
            seasonal_adjust: Whether to apply seasonal adjustment (STL if avail).
            resample_agg: Aggregation for resampling (max|mean|sum|any).
            fill_missing: Missing policy for resampled frame (zero|ffill|nan).
            max_lag_minutes: Max lag window for cross-correlation search.
            corr_method: Correlation method ('spearman' or 'pearson').
            fdr_alpha: Alpha for FDR correction (BH/BY).
            residual_include_global: Include global factor in residualization.
            residual_prior_strength: Prior strength for bucket smoothing.
            residual_variance_stabilize: Apply Bernoulli variance stabilization.
        """
        self.default_correlation_threshold: float = correlation_threshold
        self.default_jaccard_threshold: float = jaccard_threshold  # New attribute
        self.df_resampled: pd.DataFrame
        self.correlation_matrix: pd.DataFrame
        self.jaccard_matrix: pd.DataFrame
        self.user_list: List[str]
        self.resample_offset: pd.DateOffset | pd.Timedelta | None = None
        self.resample_seconds: Optional[float] = None
        self.debug_users: List[str] = debug_users or []
        # Store new options
        self.ewma_alpha: Optional[float] = (
            ewma_alpha if (ewma_alpha is None or 0 < ewma_alpha <= 1.0) else None
        )
        if ewma_alpha is not None and self.ewma_alpha is None:
            logger.warning(
                "ewma_alpha (%s) is outside valid range (0,1]. Disabling EWMA.",
                ewma_alpha,
            )
        self.seasonal_adjust: bool = bool(seasonal_adjust and _HAS_STL)
        if seasonal_adjust and not _HAS_STL:
            logger.warning(
                "seasonal_adjust requested but statsmodels not available. Skipping seasonal adjustment."
            )
        self.max_lag_minutes: int = max(0, int(max_lag_minutes))
        self.corr_method: str = corr_method
        self.fdr_alpha: float = float(fdr_alpha)
        self.fill_missing: str = (fill_missing or "zero").lower()
        if self.fill_missing not in ("zero", "ffill", "nan"):
            self.fill_missing = "zero"
        self.resample_agg: str = (resample_agg or "max").lower()
        if self.resample_agg not in ("max", "mean", "sum", "any"):
            self.resample_agg = "max"
        # residualization defaults
        self._resid_include_global: bool = bool(residual_include_global)
        self._resid_prior_strength: int = int(residual_prior_strength)
        self._resid_varstab: bool = bool(residual_variance_stabilize)
        # cache fingerprint holders
        self._fp_for_corr: str | None = None
        self._fp_resid: str | None = None

        if activity_df.empty:
            logger.warning(
                "TemporalAnalyzer initialized with an empty DataFrame."
            )
            self.df_resampled = pd.DataFrame()
            self.df_for_corr = pd.DataFrame()
            self.df_resid = None
            self.correlation_matrix = pd.DataFrame()
            self.jaccard_matrix = pd.DataFrame()  # Initialize empty
            # Initialize cross-correlation related placeholders to avoid attribute errors
            self.crosscorr_max = pd.DataFrame()
            self.crosscorr_best_lag_seconds = pd.DataFrame()
            self.crosscorr_pval = pd.DataFrame()
            self.crosscorr_qval = pd.DataFrame()
            self.user_list = []
            return

        self.user_list = activity_df.columns.tolist()
        activity_df_numeric: pd.DataFrame = activity_df.astype(float)

        # Determine resample frequency (supports 'auto'). If 'auto', we
        # estimate a reasonable bucket size from the median spacing of points.
        if (resample_period or "").lower() == "auto":
            resample_period = self._auto_resample_period(activity_df_numeric.index)
        # Resample and fill missing according to policy
        if self.resample_agg == "max" or self.resample_agg == "any":
            agg = "max"
        elif self.resample_agg == "sum":
            agg = "sum"
        else:
            agg = "mean"
        df_rs = activity_df_numeric.resample(resample_period).agg(agg)
        if self.fill_missing == "zero":
            df_rs = df_rs.fillna(0)
        elif self.fill_missing == "ffill":
            df_rs = df_rs.ffill()
        # else: "nan" keeps NaNs
        self.df_resampled = df_rs
        logger.info(
            "Activity data resampled to %s. Shape: %s",
            resample_period,
            self.df_resampled.shape,
        )

        # Optional compact debug output for selected users
        if self.debug_users:
            for dbg_user in self.debug_users:
                if dbg_user in self.df_resampled.columns:
                    series_dbg = self.df_resampled[dbg_user]
                    logger.debug(
                        "Resampled sample for %s (head5):\n%s",
                        dbg_user,
                        series_dbg.head(5).to_string(),
                    )
                    logger.debug(
                        "Resampled sample for %s (tail5):\n%s",
                        dbg_user,
                        series_dbg.tail(5).to_string(),
                    )

        # Cache frequency info once
        self._infer_resample_offset_and_seconds()  # Cache step size in seconds

        # Prepare frame for correlation (preprocessing is optional)
        self.df_for_corr: pd.DataFrame = self._prepare_corr_frame(self.df_resampled)
        # Residual frame (time-of-week/global) computed lazily on demand
        self.df_resid: pd.DataFrame | None = None
        self._tz_source: str | None = None
        self._resid_meta: Dict[str, Any] = {}

        # Core matrices
        self.correlation_matrix = self._calculate_correlations(method=self.corr_method)
        self.jaccard_matrix = self._calculate_jaccard_indices()

        # Cross-correlation scan with lag selection and FDR
        if self.max_lag_minutes > 0 and len(self.user_list) >= 2:
            (
                self.crosscorr_max,
                self.crosscorr_best_lag_seconds,
                self.crosscorr_pval,
                self.crosscorr_qval,
            ) = self._calculate_cross_correlations_with_lag(
                method=self.corr_method,
                max_lag_minutes=self.max_lag_minutes,
                fdr_alpha=self.fdr_alpha,
                fdr_method="by",
            )
        else:
            self.crosscorr_max = pd.DataFrame()
            self.crosscorr_best_lag_seconds = pd.DataFrame()
            self.crosscorr_pval = pd.DataFrame()
            self.crosscorr_qval = pd.DataFrame()

    # ---- Preprocessing for correlation ----
    def _frame_fingerprint(self, frame: pd.DataFrame | None, tag: str) -> str:
        if frame is None or frame.empty:
            return khash(tag, "empty")
        try:
            idx = frame.index
            start = str(idx[0]) if len(idx) else "na"
            end = str(idx[-1]) if len(idx) else "na"
            shape = frame.shape
            arr = frame.to_numpy(dtype=float, copy=False)
            finite = np.isfinite(arr)
            s = float(np.nansum(np.where(finite, arr, 0.0)))
            ss = float(np.nansum(np.where(finite, arr * arr, 0.0)))
            return khash(tag, shape, start, end, round(s, 6), round(ss, 6))
        except Exception:
            return khash(tag, id(frame), frame.shape)

    def _prepare_corr_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply optional seasonal adjustment and EWMA smoothing for correlations.

        Operates on a float copy to leave binary frames intact for Jaccard.
        - Seasonal adjustment via STL (daily period) if available and feasible.
        - EWMA smoothing if `ewma_alpha` set.
        """
        out = df.astype(float).copy()
        # Seasonal adjustment
        if self.seasonal_adjust and self.resample_seconds:
            daily_period = (
                int(round(24 * 3600 / float(self.resample_seconds)))
                if self.resample_seconds
                else None
            )
            if daily_period and daily_period >= 12 and len(out) >= daily_period * 2:
                for col in out.columns:
                    try:
                        stl = STL(out[col], period=daily_period, robust=True)
                        res = stl.fit()
                        out[col] = res.resid
                    except Exception:
                        out[col] = out[col] - out[col].mean()
            else:
                out = out - out.mean(axis=0)
        # EWMA smoothing
        if self.ewma_alpha is not None:
            out = out.ewm(alpha=self.ewma_alpha, adjust=False).mean()
        return out

    def _infer_resample_offset_and_seconds(self) -> None:
        """
        Infers and caches the resample offset and seconds for downstream use.
        """
        if self.df_resampled.empty:
            self.resample_offset = None
            self.resample_seconds = None
            return

        freq_offset: pd.DateOffset | None = self.df_resampled.index.freq
        if freq_offset is None:
            inferred = pd.infer_freq(self.df_resampled.index)
            if inferred:
                try:
                    freq_offset = pd.tseries.frequencies.to_offset(inferred)
                except Exception:
                    freq_offset = None

        if freq_offset is None:
            # Fallback 1 minute step if not inferable
            freq_offset = pd.Timedelta(minutes=1)
            logger.warning(
                "Could not infer resample frequency, defaulting to 1 minute."
            )

        self.resample_offset = freq_offset
        # Convert to seconds regardless of offset type
        if hasattr(freq_offset, "nanos"):
            self.resample_seconds = float(getattr(freq_offset, "nanos")) / 1e9
        else:
            self.resample_seconds = pd.to_timedelta(freq_offset).total_seconds()

    def _auto_resample_period(self, idx: pd.Index) -> str:
        """Heuristically choose resample period from timestamp gaps.

        Uses median delta across index; clamps to [60s, 900s]. Returns pandas
        offset alias like '1min', '5min'.
        """
        try:
            if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
                return "1min"
            diffs = np.diff(idx.view("i8"))  # in ns
            if diffs.size == 0:
                return "1min"
            med_ns = np.median(diffs)
            sec = max(60.0, min(900.0, float(med_ns) / 1e9))
            # round to nearest 30s
            sec = max(60.0, min(900.0, round(sec / 30.0) * 30.0))
            if sec % 60 == 0:
                return f"{int(sec//60)}min"
            else:
                return f"{int(sec)}s"
        except Exception:
            return "1min"

    # ---- Residualization (time-of-week and optional global factor) ----
    def residualize_time(
        self,
        kind: str = "onehot",
        include_global: bool = False,
        prior_strength: int = 24,
        variance_stabilize: bool = True,
        tz: Optional[str] = None,
        tz_offsets_per_user: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Fast residualization by time of week + global factor.

        Algorithm (vectorized):
        - Build bucket indices = dayofweek*24 + hour (0..167).
        - For each bucket, compute mean across all users: 168×n (loop over 168, constant).
        - Baseline at each time point — simple selection bucket_means[bucket[t], :].
        - Residuals = df - baseline. If include_global: estimate beta coefficients for
          projection onto global factor g (mean across users) vectorized and subtract g*beta.
        """
        # apply instance defaults if caller didn't override
        if include_global is False and getattr(self, "_resid_include_global", False):
            include_global = True
        if prior_strength == 24 and hasattr(self, "_resid_prior_strength"):
            prior_strength = int(self._resid_prior_strength)
        if variance_stabilize is True and hasattr(self, "_resid_varstab"):
            variance_stabilize = bool(self._resid_varstab)

        if self.df_resampled.empty:
            return pd.DataFrame()
        df = self.df_resampled.astype(float)
        idx = df.index  # Respect existing timezone; API may convert tz upstream
        if tz:
            try:
                idx = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize(tz)
                self._tz_source = tz
            except Exception:
                pass
        X = df.to_numpy(copy=False)
        T, n = X.shape
        users = list(df.columns)
        # Individual buckets per user (support for TZ offsets at user level)
        sums = np.zeros((168, n), dtype=float)
        counts = np.zeros((168, n), dtype=int)
        buckets_per_user: List[np.ndarray] = []
        for j, u in enumerate(users):
            off_h = 0.0
            try:
                if tz_offsets_per_user and u in tz_offsets_per_user:
                    off_h = float(tz_offsets_per_user[u])
            except Exception:
                off_h = 0.0
            try:
                idx_shift = idx + pd.to_timedelta(off_h, unit="h") if off_h else idx
            except Exception:
                idx_shift = idx
            bucket_j = (
                idx_shift.dayofweek.to_numpy() * 24 + idx_shift.hour.to_numpy()
            ).astype(int)
            buckets_per_user.append(bucket_j)
            counts[:, j] = np.bincount(bucket_j, minlength=168)
            sums[:, j] = np.bincount(bucket_j, weights=X[:, j], minlength=168)

        # Empirical Bayesian smoothing: p_hat[b, j] = (sum[b, j] + alpha * p0[j]) / (count[b, j] + alpha)
        alpha = max(0, int(prior_strength))
        p0 = np.clip(X.mean(axis=0), 1e-6, 1 - 1e-6)
        if alpha == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                p_hat = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
            p_hat = np.where(counts > 0, p_hat, p0[None, :])
        else:
            p_hat = (sums + alpha * p0[None, :]) / (counts + alpha)

        # Baseline at each time point by columns
        baseline = np.zeros((T, n), dtype=float)
        for j in range(n):
            baseline[:, j] = p_hat[buckets_per_user[j], j]
        resid_arr = X - baseline

        # Variance-stabilizing normalization for Bernoulli: (x - p) / sqrt(p(1-p))
        if variance_stabilize:
            denom = np.sqrt(np.clip(baseline * (1.0 - baseline), 1e-6, None))
            resid_arr = resid_arr / denom
        if include_global and n >= 2:
            g = resid_arr.mean(axis=1)  # (T,)
            gv = g
            gvar = float(np.var(gv))
            if gvar > 1e-12:
                # beta (vector of length n) = (Y^T g) / (T * var(g))
                beta = (resid_arr.T @ gv) / (len(gv) * gvar)  # (n,)
                resid_arr = resid_arr - gv[:, None] * beta[None, :]
        resid = pd.DataFrame(resid_arr, index=df.index, columns=df.columns)
        self.df_resid = resid
        try:
            self._resid_meta = {
                "include_global": include_global,
                "prior_strength": prior_strength,
                "variance_stabilize": variance_stabilize,
                "tz_source": self._tz_source,
                "tz_offsets_per_user": tz_offsets_per_user,
            }
        except Exception:
            pass
        return resid

    def _calculate_correlations(self, method: str = "spearman") -> pd.DataFrame:
        """
        Compute user-by-user correlation matrix on the prepared frame.

        Args:
            method: Correlation method ('spearman' or 'pearson').

        Returns:
            pd.DataFrame: Correlation matrix; empty if insufficient data.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            logger.warning("Not enough data or users to calculate correlations.")
            return pd.DataFrame()

        base = (
            self.df_for_corr
            if hasattr(self, "df_for_corr") and not self.df_for_corr.empty
            else self.df_resampled
        )
        corr_matrix: pd.DataFrame = base.corr(method=method)
        logger.info("Correlation matrix calculated using %s method.", method)
        return corr_matrix

    # ---- Cross-correlation with lag search and FDR ----
    def _calculate_cross_correlations_with_lag(
        self,
        method: str = "spearman",
        max_lag_minutes: int = 15,
        fdr_alpha: float = 0.05,
        fdr_method: str = "by",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Vectorized cross-correlation with lag search and FDR correction.

        What it does:
        - Scans lags (lead/lag in steps) between every pair of users.
        - For each lag, computes a full correlation matrix using overlapping
          windows only (to stay fair when shifting).
        - Picks the best absolute correlation and the corresponding lag.
        - Estimates p-values via random circular shifts (permutation idea).
        - Applies FDR to control the expected fraction of false positives.

        Returns:
        - max r: strongest correlation for each pair across lags.
        - best lag (seconds): the lag where this max r happens.
        - p-values: probability the observed or stronger result appears by
          chance under random shifts.
        - q-values: FDR-adjusted p-values (lower means more reliable).
        """
        # cache key
        try:
            fp = self._frame_fingerprint(self.df_for_corr, "for_corr")
            users = (
                list(self.df_for_corr.columns) if self.df_for_corr is not None else []
            )
            key = f"tgtrax:xcorr_scan:{khash(method, max_lag_minutes, self.resample_seconds, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["r"], index=users, columns=users, dtype=float),
                    pd.DataFrame(
                        cached["lag"], index=users, columns=users, dtype=float
                    ),
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass
        if self.df_for_corr.empty or len(self.df_for_corr.columns) < 2:
            logger.warning("Not enough data for cross-correlation.")
            empty = pd.DataFrame()
            return empty, empty, empty, empty

        # Ensure resample seconds
        if not self.resample_seconds:
            try:
                inferred = pd.infer_freq(self.df_for_corr.index)
                step = pd.tseries.frequencies.to_offset(inferred)
                seconds = (
                    float(getattr(step, "nanos", pd.to_timedelta(step).value)) / 1e9
                )
                self.resample_seconds = seconds
            except Exception:
                self.resample_seconds = 60.0
            # resample_seconds: the size of one step (bucket) in seconds, used
            # to convert best lag in steps into seconds for human readability.

        arr0 = self.df_for_corr.values.astype(float)
        if method != "pearson":
            try:
                arr = self.df_for_corr.rank(method="average").values.astype(float)
            except Exception:
                arr = arr0
        else:
            arr = arr0

        T, n = arr.shape
        users = list(self.df_for_corr.columns)
        K = max(0, int(round((max_lag_minutes * 60.0) / float(self.resample_seconds))))

        best_abs = np.full((n, n), -1.0, dtype=float)
        best_r = np.zeros((n, n), dtype=float)
        best_k = np.zeros((n, n), dtype=int)
        best_L = np.zeros((n, n), dtype=int)

        iu = np.triu_indices(n, k=1)

        def corr_matrix_xy(
            X: np.ndarray, Y: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            L = X.shape[0]
            sx = X.sum(axis=0)
            sy = Y.sum(axis=0)
            ssx = np.einsum("ij,ij->j", X, X)
            ssy = np.einsum("ij,ij->j", Y, Y)
            sxy = X.T @ Y
            num = sxy - np.outer(sx, sy) / L
            vx = ssx - (sx * sx) / L
            vy = ssy - (sy * sy) / L
            with np.errstate(invalid="ignore", divide="ignore"):
                R = num / np.sqrt(np.outer(vx, vy))
            return R, vx, vy

        # Try all lags from -K to +K. Negative: Y leads X. Positive: X leads Y.
        # We always use overlapping segments only (no padding) to keep fairness.
        for k in range(-K, K + 1):
            if k == 0:
                L = T
                X = arr
                Y = arr
            elif k > 0:
                L = T - k
                if L < 3:
                    continue
                X = arr[k:, :]
                Y = arr[:L, :]
            else:
                kk = -k
                L = T - kk
                if L < 3:
                    continue
                X = arr[:L, :]
                Y = arr[kk:, :]
            R, vx, vy = corr_matrix_xy(X, Y)
            R_abs = np.abs(R[iu])
            mask = R_abs > best_abs[iu]
            if np.any(mask):
                idx_i = iu[0][mask]
                idx_j = iu[1][mask]
                best_abs[idx_i, idx_j] = R_abs[mask]
                best_r[idx_i, idx_j] = R[iu][mask]
                best_k[idx_i, idx_j] = k
                best_L[idx_i, idx_j] = L

        # Mirror to full matrix & lags
        best_r = best_r + best_r.T
        # Build antisymmetric lag matrix explicitly: lag[j,i] = -lag[i,j]
        lag_seconds = np.zeros((n, n), dtype=float)
        lag_vals = best_k[iu] * float(self.resample_seconds)
        lag_seconds[iu] = lag_vals
        lag_seconds[(iu[1], iu[0])] = -lag_vals

        # p-values via t-distribution (approx.)
        pvals = np.full((n, n), np.nan, dtype=float)
        if student_t is not None:
            r_vec = best_r[iu]
            L_vec = best_L[iu].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stat = r_vec * np.sqrt(
                    np.maximum(L_vec - 2.0, 1.0)
                    / np.maximum(1e-12, 1.0 - r_vec * r_vec)
                )
            df = np.maximum(L_vec - 2.0, 1.0)
            p_vec = 2.0 * student_t.sf(np.abs(t_stat), df)
            # Multiple-lag scan correction (per-pair Bonferroni)
            try:
                lag_count = int(2 * K + 1)
            except Exception:
                lag_count = 1
            if lag_count > 1:
                p_vec = np.minimum(1.0, p_vec * float(lag_count))
            pvals[iu] = p_vec
            pvals = pvals + pvals.T
            # Use NaN on diagonal to avoid misinterpretation downstream
            np.fill_diagonal(pvals, np.nan)

        pvec = pvals[iu]
        if (fdr_method or "by").lower() == "by":
            qvec = self._by_fdr(pvec)
        else:
            qvec = self._bh_fdr(pvec)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        qvals[iu] = qvec
        qvals = qvals + qvals.T
        np.fill_diagonal(qvals, np.nan)

        logger.info(
            "Cross-correlation computed (vectorized) with lag window ±%s min.",
            max_lag_minutes,
        )
        out = (
            pd.DataFrame(best_r, index=users, columns=users, dtype=float),
            pd.DataFrame(lag_seconds, index=users, columns=users, dtype=float),
            pd.DataFrame(pvals, index=users, columns=users, dtype=float),
            pd.DataFrame(qvals, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(
                key,
                {"r": best_r, "lag": lag_seconds, "p": pvals, "q": qvals},
                ttl=default_ttl("heavy"),
            )
        except Exception:
            pass
        return out

    def compute_crosscorr_with_lag(
        self,
        method: str = "spearman",
        max_lag_minutes: Optional[int] = None,
        residualize: bool = False,
        fdr_method: str = "bh",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute cross-correlation with lag search on-demand from selected frame.

        This mirrors _calculate_cross_correlations_with_lag but allows choosing
        residualized series when residualize=True.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            empty = pd.DataFrame()
            return empty, empty, empty, empty
        frame = (
            self.df_resid
            if (residualize and self.df_resid is not None)
            else self.df_for_corr
        )
        if frame is None or frame.empty:
            empty = pd.DataFrame()
            return empty, empty, empty, empty
        # Ensure resample seconds
        if not self.resample_seconds:
            try:
                inferred = pd.infer_freq(frame.index)
                step = pd.tseries.frequencies.to_offset(inferred)
                seconds = (
                    float(getattr(step, "nanos", pd.to_timedelta(step).value)) / 1e9
                )
                self.resample_seconds = seconds
            except Exception:
                self.resample_seconds = 60.0

        # cache
        try:
            fp = self._frame_fingerprint(frame, "resid" if residualize else "for_corr")
            users = list(frame.columns)
            Kmin = self.max_lag_minutes if max_lag_minutes is None else max_lag_minutes
            key = f"tgtrax:xcorr_on_demand:{khash(method, Kmin, residualize, fdr_method, self.resample_seconds, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["r"], index=users, columns=users, dtype=float),
                    pd.DataFrame(
                        cached["lag"], index=users, columns=users, dtype=float
                    ),
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass

        arr0 = frame.values.astype(float)
        if method != "pearson":
            try:
                arr = frame.rank(method="average").values.astype(float)
            except Exception:
                arr = arr0
        else:
            arr = arr0

        T, n = arr.shape
        users = list(frame.columns)
        K = max(
            0,
            int(
                round(
                    (
                        (
                            self.max_lag_minutes
                            if max_lag_minutes is None
                            else max_lag_minutes
                        )
                        * 60.0
                    )
                    / float(self.resample_seconds)
                )
            ),
        )

        best_abs = np.full((n, n), -1.0, dtype=float)
        best_r = np.zeros((n, n), dtype=float)
        best_k = np.zeros((n, n), dtype=int)
        best_L = np.zeros((n, n), dtype=int)
        iu = np.triu_indices(n, k=1)

        def corr_matrix_xy(
            X: np.ndarray, Y: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            L = X.shape[0]
            sx = X.sum(axis=0)
            sy = Y.sum(axis=0)
            ssx = np.einsum("ij,ij->j", X, X)
            ssy = np.einsum("ij,ij->j", Y, Y)
            sxy = X.T @ Y
            num = sxy - np.outer(sx, sy) / L
            vx = ssx - (sx * sx) / L
            vy = ssy - (sy * sy) / L
            with np.errstate(invalid="ignore", divide="ignore"):
                R = num / np.sqrt(np.outer(vx, vy))
            return R, vx, vy

        for k in range(-K, K + 1):
            if k == 0:
                L = T
                X = arr
                Y = arr
            elif k > 0:
                L = T - k
                if L < 3:
                    continue
                X = arr[k:, :]
                Y = arr[:L, :]
            else:
                kk = -k
                L = T - kk
                if L < 3:
                    continue
                X = arr[:L, :]
                Y = arr[kk:, :]
            R, vx, vy = corr_matrix_xy(X, Y)
            R_abs = np.abs(R[iu])
            mask = R_abs > best_abs[iu]
            if np.any(mask):
                idx_i = iu[0][mask]
                idx_j = iu[1][mask]
                best_abs[idx_i, idx_j] = R_abs[mask]
                best_r[idx_i, idx_j] = R[iu][mask]
                best_k[idx_i, idx_j] = k
                best_L[idx_i, idx_j] = L

        best_r = best_r + best_r.T
        lag_seconds = np.zeros((n, n), dtype=float)
        lag_vals = best_k[iu] * float(self.resample_seconds)
        lag_seconds[iu] = lag_vals
        lag_seconds[(iu[1], iu[0])] = -lag_vals

        pvals = np.full((n, n), np.nan, dtype=float)
        if student_t is not None:
            r_vec = best_r[iu]
            L_vec = best_L[iu].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stat = r_vec * np.sqrt(
                    np.maximum(L_vec - 2.0, 1.0)
                    / np.maximum(1e-12, 1.0 - r_vec * r_vec)
                )
            df = np.maximum(L_vec - 2.0, 1.0)
            p_vec = 2.0 * student_t.sf(np.abs(t_stat), df)
            # Multiple lag selection correction (per-pair Bonferroni/Šidák)
            # lags tested: 2K+1
            try:
                lag_count = int(2 * K + 1)
            except Exception:
                lag_count = 1
            if lag_count > 1:
                p_vec = np.minimum(1.0, p_vec * float(lag_count))
            pvals[iu] = p_vec
            pvals = pvals + pvals.T
            np.fill_diagonal(pvals, np.nan)
        pvec = pvals[iu]
        if (fdr_method or "bh").lower() == "by":
            qvec = self._by_fdr(pvec)
        else:
            qvec = self._bh_fdr(pvec)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        qvals[iu] = qvec
        qvals = qvals + qvals.T
        np.fill_diagonal(qvals, np.nan)

        out = (
            pd.DataFrame(best_r, index=users, columns=users, dtype=float),
            pd.DataFrame(lag_seconds, index=users, columns=users, dtype=float),
            pd.DataFrame(pvals, index=users, columns=users, dtype=float),
            pd.DataFrame(qvals, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(
                key,
                {"r": best_r, "lag": lag_seconds, "p": pvals, "q": qvals},
                ttl=default_ttl("heavy"),
            )
        except Exception:
            pass
        return out

    # ---- Permutation p-values for max-lag cross-correlation ----
    def crosscorr_pvals_max_lag_perm(
        self,
        perms: int = 200,
        max_lag_minutes: Optional[int] = None,
        method: str = "spearman",
        use_residuals: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        batch_size: int = 16,
        fdr_method: str = "by",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute permutation p-values for the max-|r| across lags statistic for
        each user pair using circular shifts to preserve autocorrelation.

        Returns (pvals_df, qvals_df). For small T or insufficient users, returns
        empty DataFrames.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            return pd.DataFrame(), pd.DataFrame()
        if perms <= 0:
            return pd.DataFrame(), pd.DataFrame()
        users = self.user_list
        T = len(self.df_resampled.index)
        if T < 4:
            return pd.DataFrame(), pd.DataFrame()
        max_lag = int(
            self.max_lag_minutes if max_lag_minutes is None else max_lag_minutes
        )
        if max_lag <= 0 or self.resample_seconds is None:
            return pd.DataFrame(), pd.DataFrame()
        max_lag_steps = max(
            1, int(round(max_lag * 60.0 / float(self.resample_seconds)))
        )

        # cache
        try:
            frame = (
                self.df_resid
                if (use_residuals and self.df_resid is not None)
                else self.df_for_corr
            )
            users = list(frame.columns) if frame is not None else []
            fp = self._frame_fingerprint(
                frame, "resid" if use_residuals else "for_corr"
            )
            key = f"tgtrax:xcorr_perm:{khash(method, perms, max_lag, use_residuals, random_state, n_jobs, batch_size, fdr_method, self.resample_seconds, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass

        rng = np.random.default_rng(random_state)
        frame = (
            self.df_resid
            if (use_residuals and self.df_resid is not None)
            else self.df_for_corr
        ).astype(float)
        arr0 = frame.values  # (T, n)
        # Spearman as ranks + Pearson (faster):
        if method != "pearson":
            try:
                arr = frame.rank(method="average").values.astype(float)
            except Exception:
                arr = arr0.astype(float)
        else:
            arr = arr0.astype(float)
        n = arr.shape[1]

        # Observed maximum |r|
        obs_max = np.full((n, n), np.nan, dtype=float)
        if (
            not getattr(self, "crosscorr_max", pd.DataFrame()).empty
            and list(self.crosscorr_max.columns) == users
        ):
            obs_mat = self.crosscorr_max.values
            obs_max[:, :] = np.abs(obs_mat)
        else:
            sx = arr.sum(axis=0)
            ss = np.einsum("ij,ij->j", arr, arr)
            num = arr.T @ arr - np.outer(sx, sx) / T
            var = ss - (sx * sx) / T
            with np.errstate(invalid="ignore", divide="ignore"):
                R0 = num / np.sqrt(np.outer(var, var))
            obs_max = np.abs(R0)

        # Parallel processing of permutations in batches
        shifts = rng.integers(low=0, high=T, size=perms, dtype=np.int64)
        if batch_size < 1:
            batch_size = 16
        batches = [
            shifts[i : i + batch_size] for i in range(0, len(shifts), batch_size)
        ]
        if n_jobs is None:
            cpu = os.cpu_count() or 2
            n_jobs = max(1, min(4, cpu - 1))

        ge_counts = np.zeros((n, n), dtype=float)
        if n_jobs == 1 or len(batches) == 1:
            for b in batches:
                ge_counts += _xcorr_perm_worker_batch(
                    arr, T, n, max_lag_steps, obs_max, b
                )
        else:
            try:
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                try:
                    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                    shm_arr[:] = arr
                    with ProcessPoolExecutor(
                        max_workers=n_jobs,
                        initializer=_init_worker_shm,
                        initargs=(shm.name, arr.shape, str(arr.dtype)),
                    ) as ex:
                        futures = [
                            ex.submit(
                                _xcorr_perm_worker_batch_shm,
                                T,
                                n,
                                max_lag_steps,
                                obs_max,
                                b,
                            )
                            for b in batches
                        ]
                        for f in as_completed(futures):
                            ge_counts += f.result()
                finally:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
            except Exception:
                # Fallback to single-threaded processing if multiprocessing fails
                for b in batches:
                    ge_counts += _xcorr_perm_worker_batch(
                        arr, T, n, max_lag_steps, obs_max, b
                    )
        pvals = (ge_counts + 1.0) / float(perms + 1)
        np.fill_diagonal(pvals, np.nan)
        iu = np.triu_indices(n, k=1)
        pvec = pvals[iu]
        if (fdr_method or "bh").lower() == "by":
            qvec = self._by_fdr(pvec)
        else:
            qvec = self._bh_fdr(pvec)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        qvals[iu] = qvec
        qvals = qvals + qvals.T
        np.fill_diagonal(qvals, np.nan)
        out = (
            pd.DataFrame(pvals, index=users, columns=users, dtype=float),
            pd.DataFrame(qvals, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(key, {"p": pvals, "q": qvals}, ttl=default_ttl("heavy"))
        except Exception:
            pass
        return out

    @staticmethod
    def _bh_fdr(pvalues_1d: np.ndarray) -> np.ndarray:
        """Benjamini–Hochberg FDR control. Returns q-values for given p-values.
        NaNs preserved; order preserved.
        """
        p = pvalues_1d.copy()
        idx = np.where(~np.isnan(p))[0]
        if idx.size == 0:
            return p
        ps = p[idx]
        order = np.argsort(ps)
        ranked = ps[order]
        m = float(len(ranked))
        ranks = np.arange(1, len(ranked) + 1)
        q = ranked * m / ranks
        # Enforce monotone decreasing when traversed from end
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.full_like(p, np.nan, dtype=float)
        out[idx[order]] = q
        return out

    @staticmethod
    def _by_fdr(pvalues_1d: np.ndarray) -> np.ndarray:
        """Benjamini–Yekutieli FDR control (robust under dependence).

        q_i = p_(i) * m * c(m) / i, with c(m) = sum_{k=1..m} 1/k.
        Enforces monotonicity from the end. NaNs preserved; order preserved.
        """
        p = pvalues_1d.copy()
        idx = np.where(~np.isnan(p))[0]
        if idx.size == 0:
            return p
        ps = p[idx]
        order = np.argsort(ps)
        ranked = ps[order]
        m = float(len(ranked))
        c_m = np.sum(1.0 / np.arange(1.0, m + 1.0)) if m > 1 else 1.0
        ranks = np.arange(1, len(ranked) + 1)
        q = ranked * m * c_m / ranks
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.full_like(p, np.nan, dtype=float)
        out[idx[order]] = q
        return out

    # Accessors for new matrices
    def get_crosscorr_max(self) -> pd.DataFrame:
        return self.crosscorr_max

    def get_crosscorr_best_lag_seconds(self) -> pd.DataFrame:
        return self.crosscorr_best_lag_seconds

    def get_crosscorr_pvals(self) -> pd.DataFrame:
        return self.crosscorr_pval

    def get_crosscorr_qvals(self) -> pd.DataFrame:
        return self.crosscorr_qval

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Retrieves the calculated correlation matrix.

        Returns:
            A pandas DataFrame with user-to-user correlations.
        """
        return self.correlation_matrix

    # ---- Transfer Entropy (binary, k=l=1) with stationary bootstrap null ----
    @staticmethod
    def _quantize_binary(frame: pd.DataFrame, method: str = "balanced") -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        X = frame.copy()
        if method == "zero":
            return (X > 0).astype(int)
        if method.startswith("quantile:"):
            try:
                q = float(method.split(":", 1)[1])
            except Exception:
                q = 0.5
            thr = X.quantile(q, axis=0, numeric_only=True)
            for c in X.columns:
                X[c] = (X[c].to_numpy(dtype=float) > float(thr.get(c, 0.0))).astype(int)
            return X.astype(int)
        if method == "balanced":
            # Per-series quantile with guardrails on event proportion
            # Aim for ~50% ones but clip to [0.2, 0.8]
            q = 0.5
            min_p, max_p = 0.2, 0.8
            for c in X.columns:
                s = X[c].to_numpy(dtype=float)
                if np.all(~np.isfinite(s)):
                    X[c] = 0
                    continue
                # base threshold at median
                t = float(np.nanmedian(s))
                b = (s > t).astype(int)
                p = float(np.nanmean(b)) if b.size else 0.0
                if p < min_p or p > max_p:
                    # adapt quantile toward achieving min/max bounds
                    # coarse search over quantiles
                    qs = np.linspace(0.2, 0.8, 7)
                    best_q = q
                    best_diff = 1e9
                    best_b = b
                    for qq in qs:
                        tt = float(np.nanquantile(s, qq))
                        bb = (s > tt).astype(int)
                        pp = float(np.nanmean(bb)) if bb.size else 0.0
                        diff = min(abs(pp - 0.5), abs(pp - min_p), abs(pp - max_p))
                        if diff < best_diff and (min_p <= pp <= max_p):
                            best_diff = diff
                            best_q = qq
                            best_b = bb
                    b = best_b
                X[c] = b
            return X.astype(int)
        if method == "global_median":
            gm = float(np.nanmedian(X.to_numpy(dtype=float)))
            return (X.to_numpy(dtype=float) > gm).astype(int)
        if method == "global_mean":
            gm = float(np.nanmean(X.to_numpy(dtype=float)))
            return (X.to_numpy(dtype=float) > gm).astype(int)
        # default legacy: per-series median
        med = X.median(axis=0, skipna=True)
        for c in X.columns:
            X[c] = (X[c].to_numpy() > float(med.get(c, 0.0))).astype(int)
        return X.astype(int)

    @staticmethod
    def _te_binary_k1(xi: np.ndarray, xj: np.ndarray) -> float:
        T = min(len(xi), len(xj))
        if T < 3:
            return 0.0
        xi = xi.astype(int)
        xj = xj.astype(int)
        a = xj[1:T]
        b = xj[0 : T - 1]
        c = xi[0 : T - 1]
        N = np.zeros((2, 2, 2), dtype=float)
        for k in range(T - 1):
            N[a[k], b[k], c[k]] += 1.0
        N += 1.0
        total = np.sum(N)
        Pabc = N / total
        Nb_c = np.sum(N, axis=0)
        Nab_ = np.sum(N, axis=2)
        Nb__ = np.sum(Nab_, axis=0)
        Pa_bc = N / Nb_c[None, :, :]
        Pa_b = Nab_ / Nb__[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = Pa_bc / Pa_b[:, None, :]
            log_ratio = np.log(ratio)
        term = Pabc * np.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        te = float(np.sum(term))
        return max(te, 0.0)

    # If Numba is available, JIT-compile the inner TE kernel for speed
    if _HAS_NUMBA:
        _te_binary_k1 = njit(_te_binary_k1)  # type: ignore

    @staticmethod
    def _stationary_bootstrap_1d(
        x: np.ndarray, p: float = 0.1, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = len(x)
        if T <= 1:
            return x.copy()
        out = np.empty(T, dtype=x.dtype)
        i = int(rng.integers(0, T))
        out[0] = x[i]
        for t in range(1, T):
            if rng.random() < p:
                i = int(rng.integers(0, T))
            else:
                i = (i + 1) % T
            out[t] = x[i]
        return out

    def compute_transfer_entropy(
        self,
        residualize: bool = True,
        quantize: str = "balanced",
        perms: int = 200,
        bootstrap: str = "stationary",
        block_p: float = 0.1,
        random_state: Optional[int] = None,
        fdr_method: str = "by",
        early_stop: bool = True,
        alpha_stop: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute pairwise Transfer Entropy TE(i->j) with simple binary k=1 estimator.

        Removes weekly rhythm via residualize_time() when residualize=True.
        Quantizes each series to binary by median/zero threshold.
        Null via stationary bootstrap (default) of source xi; target xj fixed.
        Returns (TE, pvals, qvals) DataFrames.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        frame = (
            self.df_resid
            if (residualize and self.df_resid is not None)
            else self.df_resampled.astype(float)
        )
        if frame is None or frame.empty:
            frame = (
                self.residualize_time()
                if residualize
                else self.df_resampled.astype(float)
            )
        binf = self._quantize_binary(frame, method=quantize)
        users = list(binf.columns)
        n = len(users)
        # cache
        try:
            fp = self._frame_fingerprint(
                binf, f"te_{quantize}_{'resid' if residualize else 'raw'}"
            )
            key = f"tgtrax:te:{khash(perms, bootstrap, block_p, random_state, fdr_method, early_stop, alpha_stop, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["te"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass
        TE = np.full((n, n), np.nan, dtype=float)
        P = np.full((n, n), np.nan, dtype=float)
        rng = np.random.default_rng(random_state)
        alpha_thr = float(self.fdr_alpha if alpha_stop is None else alpha_stop)
        for i in range(n):
            xi = binf.iloc[:, i].to_numpy()
            for j in range(n):
                if i == j:
                    continue
                xj = binf.iloc[:, j].to_numpy()
                te_obs = self._te_binary_k1(xi, xj)
                if perms <= 0:
                    TE[i, j] = te_obs
                    P[i, j] = np.nan
                    continue
                ge = 0
                for b in range(perms):
                    if bootstrap == "shift":
                        s = int(rng.integers(0, len(xi)))
                        xi_s = np.roll(xi, s)
                    else:
                        xi_s = self._stationary_bootstrap_1d(
                            xi, p=max(min(block_p, 0.99), 0.01), rng=rng
                        )
                    te_b = self._te_binary_k1(xi_s, xj)
                    if te_b >= te_obs:
                        ge += 1
                    if early_stop and (b + 1) >= 50:
                        # Clopper–Pearson CI; stop if entirely above/below alpha_thr
                        if beta_dist is not None:
                            a = 0.05 / 2.0
                            lo = (
                                float(beta_dist.ppf(a, max(ge, 0), (b + 1) - ge + 1))
                                if ge > 0
                                else 0.0
                            )
                            hi = (
                                float(
                                    beta_dist.ppf(1 - a, ge + 1, max((b + 1) - ge, 0))
                                )
                                if ge < (b + 1)
                                else 1.0
                            )
                        else:
                            ph = (ge + 1.0) / (b + 3.0)
                            w = 1.96 * np.sqrt(
                                max(ph * (1 - ph) / max(b + 1, 1), 1e-12)
                            )
                            lo, hi = max(0.0, ph - w), min(1.0, ph + w)
                        if hi < alpha_thr or lo > alpha_thr:
                            perms = b + 1
                            break
                pval = (ge + 1.0) / float(perms + 1.0)
                TE[i, j] = te_obs
                P[i, j] = pval
        # Directed FDR over off-diagonal entries; do not symmetrize
        mask = (~np.eye(n, dtype=bool)) & (~np.isnan(P))
        pvec = P[mask]
        if pvec.size > 0:
            qvec = (
                self._by_fdr(pvec)
                if (fdr_method or "by").lower() == "by"
                else self._bh_fdr(pvec)
            )
            Q = np.full_like(P, np.nan, dtype=float)
            Q[mask] = qvec
        else:
            Q = np.full_like(P, np.nan, dtype=float)
        out = (
            pd.DataFrame(TE, index=users, columns=users, dtype=float),
            pd.DataFrame(P, index=users, columns=users, dtype=float),
            pd.DataFrame(Q, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(key, {"te": TE, "p": P, "q": Q}, ttl=default_ttl("heavy"))
        except Exception:
            pass
        return out

    # ---- Change-points: binary segmentation (CUSUM-like) + co-CP + NBS ----
    def _detect_changepoints_series(
        self,
        x: np.ndarray,
        max_chg: int = 8,
        min_seg_len: int = 12,
        z_thr: float = 3.0,
    ) -> List[int]:
        cps: List[int] = []
        T = len(x)
        if T < 2 * min_seg_len:
            return cps

        def rec(a: int, b: int, depth: int) -> None:
            if depth >= max_chg:
                return
            n = b - a
            if n < 2 * min_seg_len:
                return
            seg = x[a:b].astype(float)
            mu = float(seg.mean())
            sd = float(seg.std())
            if sd <= 1e-8:
                return
            cs = np.cumsum((seg - mu) / sd)
            k = int(np.argmax(np.abs(cs)))
            stat = float(np.abs(cs[k]) / np.sqrt(max(n, 1)))
            if stat < z_thr:
                return
            cp = a + k + 1
            if (cp - a) >= min_seg_len and (b - cp) >= min_seg_len:
                cps.append(cp)
                rec(a, cp, depth + 1)
                rec(cp, b, depth + 1)

        rec(0, T, 0)
        cps_sorted = sorted(set(cps))
        return cps_sorted

    def detect_changepoints(
        self,
        residualize: bool = True,
        max_chg: int = 8,
        min_seg_len: int = 12,
        z_thr: float = 3.0,
    ) -> Dict[str, List[int]]:
        frame = (
            self.df_resid
            if (residualize and getattr(self, "df_resid", None) is not None)
            else self.df_resampled.astype(float)
        )
        if frame is None or frame.empty or frame.shape[1] < 1:
            return {}
        cps: Dict[str, List[int]] = {}
        for col in frame.columns:
            x = frame[col].to_numpy()
            cps[col] = self._detect_changepoints_series(
                x, max_chg=max_chg, min_seg_len=min_seg_len, z_thr=z_thr
            )
        return cps

    def co_changepoints_matrix(
        self,
        residualize: bool = True,
        tau_steps: int = 3,
        perms: int = 200,
        fdr_method: str = "by",
        max_chg: int = 8,
        min_seg_len: int = 12,
        z_thr: float = 3.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        cps = self.detect_changepoints(
            residualize=residualize,
            max_chg=max_chg,
            min_seg_len=min_seg_len,
            z_thr=z_thr,
        )
        users = [u for u in self.user_list if u in cps]
        n = len(users)
        if n < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        T = len(self.df_resampled.index)
        A = np.zeros((n, n), dtype=float)
        P = np.full((n, n), np.nan, dtype=float)

        # helper for matches within tolerance
        def match_count(a: np.ndarray, b: np.ndarray, tau: int) -> int:
            """Greedy 1-1 matching count within tolerance tau using two pointers.

            Increments both pointers on match to avoid reusing change-points.
            Assumes a and b are sorted.
            """
            if a.size == 0 or b.size == 0:
                return 0
            i = 0
            j = 0
            cnt = 0
            while i < a.size and j < b.size:
                ai = int(a[i])
                bj = int(b[j])
                if abs(ai - bj) <= tau:
                    cnt += 1
                    i += 1
                    j += 1
                elif ai < bj - tau:
                    i += 1
                elif bj < ai - tau:
                    j += 1
                else:
                    # overlapping ranges case; move the smaller one
                    if ai < bj:
                        i += 1
                    else:
                        j += 1
            return cnt

        # observed co-CP ratio: matches / min(len(ci),len(cj))
        cp_arr = [np.array(cps[u], dtype=int) for u in users]
        for i in range(n):
            for j in range(i + 1, n):
                a = cp_arr[i]
                b = cp_arr[j]
                m = match_count(a, b, tau_steps)
                denom = max(1, min(len(a), len(b)))
                val = float(m) / float(denom)
                A[i, j] = A[j, i] = val
        # permutation p-values via circular shift of one side
        rng = np.random.default_rng()
        for i in range(n):
            for j in range(i + 1, n):
                a = cp_arr[i]
                b = cp_arr[j]
                if len(a) == 0 or len(b) == 0:
                    P[i, j] = P[j, i] = np.nan
                    continue
                obs = A[i, j]
                ge = 0
                for _ in range(max(1, int(perms))):
                    s = int(rng.integers(0, T))
                    b_s = (b + s) % T
                    b_s.sort()
                    m = match_count(a, b_s, tau_steps)
                    denom = max(1, min(len(a), len(b)))
                    val = float(m) / float(denom)
                    if val >= obs:
                        ge += 1
                p = (ge + 1.0) / float(perms + 1.0)
                P[i, j] = P[j, i] = p
        iu = np.triu_indices(n, k=1)
        pvec = P[iu]
        if (fdr_method or "by").lower() == "by":
            qvec = self._by_fdr(pvec)
        else:
            qvec = self._bh_fdr(pvec)
        Q = np.full_like(P, np.nan, dtype=float)
        Q[iu] = qvec
        Q = Q + Q.T
        return (
            pd.DataFrame(A, index=users, columns=users, dtype=float),
            pd.DataFrame(P, index=users, columns=users, dtype=float),
            pd.DataFrame(Q, index=users, columns=users, dtype=float),
        )

    def nbs_component_pvalue_from_cp(
        self,
        co_cp: pd.DataFrame,
        pvals: pd.DataFrame,
        tau_steps: int = 3,
        alpha: float = 0.05,
        perms: int = 200,
    ) -> float:
        # Build adjacency from co-cp with p<=alpha
        if co_cp is None or co_cp.empty or pvals is None or pvals.empty:
            return float("nan")
        users = list(co_cp.columns)
        n = len(users)
        import networkx as nx

        G = nx.Graph()
        for u in users:
            G.add_node(u)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = users[i], users[j]
                try:
                    p = float(pvals.iat[i, j])
                except Exception:
                    p = float("nan")
                if not np.isnan(p) and p <= alpha:
                    G.add_edge(u, v, w=float(co_cp.iat[i, j]))

        def comp_stat(graph: nx.Graph) -> float:
            if graph.number_of_edges() == 0:
                return 0.0
            comps = (graph.subgraph(c).copy() for c in nx.connected_components(graph))
            best = 0.0
            for H in comps:
                w = 0.0
                for _, _, d in H.edges(data=True):
                    w += float(d.get("w", 1.0))
                if w > best:
                    best = w
            return best

        obs = comp_stat(G)
        # Null via random rewiring preserving degree (fast heuristic)
        rng = np.random.default_rng()
        deg = dict(G.degree())
        nodes = users
        ge = 0
        for _ in range(max(1, int(perms))):
            H = nx.Graph()
            for u in nodes:
                H.add_node(u)
            stubs = []
            for u, d in deg.items():
                stubs.extend([u] * d)
            rng.shuffle(stubs)
            # pair stubs sequentially, avoid self-loops if possible
            for k in range(0, len(stubs) - 1, 2):
                a, b = stubs[k], stubs[k + 1]
                if a == b:
                    continue
                # pick random weight from observed edges distribution
                try:
                    w = float(co_cp.sample(1).values[0][0])
                except Exception:
                    w = 1.0
                H.add_edge(a, b, w=w)
            if comp_stat(H) >= obs:
                ge += 1
        p = (ge + 1.0) / float(perms + 1.0)
        return float(p)

    # ---- Hawkes-like excitation score (directed) with circular-shift null ----
    def _binary_event_impulses(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Return (impulses T×n, event_indices per user) for start-of-online events.

        Rising edges: x[t-1]=0 → x[t]=1. Works on resampled binary frame.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 1:
            return np.zeros((0, 0), dtype=np.int8), []
        X = self._binary_frame().to_numpy(dtype=np.int8, copy=False)
        T, n = X.shape
        if T < 2:
            return np.zeros((T, n), dtype=np.int8), [
                np.array([], dtype=int) for _ in range(n)
            ]
        prev = np.vstack([np.zeros((1, n), dtype=np.int8), X[:-1, :]])
        starts = ((X == 1) & (prev == 0)).astype(np.int8)
        ev_idx = [np.where(starts[:, j] == 1)[0].astype(int) for j in range(n)]
        return starts, ev_idx

    def hawkes_excitation(
        self,
        half_life_minutes: float = 10.0,
        max_lag_minutes: Optional[float] = None,
        perms: int = 200,
        random_state: Optional[int] = None,
        fdr_method: str = "by",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute directed Hawkes-like excitation score S_ij and its p/q-values.

        S_ij = average over j-start events of sum_{t_i < t_j} exp(-beta * (t_j - t_i)).
        Implementation via 1D causal convolution of i's start-impulse with kernel
        k[u] = exp(-beta * u*dt) for u=1..K (k[0]=0), sampled at j event indices.
        Null via circular shift of i's series (permute source only).
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        dt = float(self.resample_seconds or 60.0)
        starts, ev_idx = self._binary_event_impulses()
        T, n = starts.shape
        if T < 3:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        users = list(self.df_resampled.columns)
        # cache
        try:
            fp = self._frame_fingerprint(self.df_resampled, "resampled")
            key = f"tgtrax:hawkes:{khash(half_life_minutes, max_lag_minutes, perms, random_state, fdr_method, self.resample_seconds, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["S"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["P"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["Q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass
        # kernel
        beta = np.log(2.0) / max(1e-6, (half_life_minutes * 60.0))
        if max_lag_minutes is None:
            max_lag_minutes = min(60.0, max(5.0, self.max_lag_minutes))
        K = max(1, int(round((max_lag_minutes * 60.0) / dt)))
        u = np.arange(K + 1, dtype=float)  # 0..K
        k = np.exp(-beta * u * dt)
        k[0] = 0.0  # exclude instantaneous effect

        # Precompute convolutions for each source i
        convs: List[np.ndarray] = []
        for i in range(n):
            sig = starts[:, i].astype(float)
            if _HAS_FFTCONV and T > 1024:
                ci = fftconvolve(sig, k, mode="full")[:T]
            else:
                ci = np.convolve(sig, k, mode="full")[:T]
            convs.append(ci)

        # Observed S_ij
        S = np.full((n, n), np.nan, dtype=float)
        for i in range(n):
            ci = convs[i]
            for j in range(n):
                if i == j:
                    continue
                idx = ev_idx[j]
                if idx.size == 0:
                    continue
                S[i, j] = float(np.mean(ci[idx]))

        # Permutation p-values (shift source series)
        rng = np.random.default_rng(random_state)
        ge = np.zeros((n, n), dtype=float)
        if perms > 0:
            shifts = rng.integers(low=0, high=T, size=int(perms), dtype=np.int64)
            for s in shifts:
                for i in range(n):
                    ci = np.roll(convs[i], int(s))
                    for j in range(n):
                        if i == j:
                            continue
                        idx = ev_idx[j]
                        if idx.size == 0:
                            continue
                        val = float(np.mean(ci[idx]))
                        if not np.isnan(S[i, j]) and val >= S[i, j]:
                            ge[i, j] += 1.0
        P = (ge + 1.0) / float(perms + 1.0) if perms > 0 else np.full_like(S, np.nan)
        np.fill_diagonal(P, np.nan)

        # FDR on upper triangle of directed matrix? We apply on all i!=j pairs treating them as a 1D list
        mask = (~np.eye(n, dtype=bool)) & (~np.isnan(P))
        pvec = P[mask]
        if pvec.size > 0:
            qvec = (
                self._by_fdr(pvec)
                if (fdr_method or "by").lower() == "by"
                else self._bh_fdr(pvec)
            )
            Q = np.full_like(P, np.nan, dtype=float)
            Q[mask] = qvec
        else:
            Q = np.full_like(P, np.nan, dtype=float)

        out = (
            pd.DataFrame(S, index=users, columns=users, dtype=float),
            pd.DataFrame(P, index=users, columns=users, dtype=float),
            pd.DataFrame(Q, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(key, {"S": S, "P": P, "Q": Q}, ttl=default_ttl("heavy"))
        except Exception:
            pass
        return out

    def compute_correlation_matrix(
        self, method: str = "spearman", residualize: bool = False
    ) -> pd.DataFrame:
        """Compute correlation matrix from selected frame on demand.
        If residualize=True and residuals available/possible, use them.
        """
        # Ensure residuals if requested
        frame: pd.DataFrame | None
        if residualize:
            # extra guard in case legacy instances missed df_resid
            df_resid_local = getattr(self, "df_resid", None)
            frame = (
                df_resid_local
                if df_resid_local is not None
                else self.residualize_time()
            )
        else:
            frame = self.df_for_corr
        if frame is None or frame.empty:
            return pd.DataFrame()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                if method == "pearson":
                    return frame.corr(method="pearson")
                else:
                    return frame.corr(method="spearman")
        except Exception:
            return pd.DataFrame()

    def compute_correlation_pq(
        self,
        method: str = "spearman",
        residualize: bool = False,
        fdr_method: str = "by",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute correlation, p-values and q-values for pairwise correlations.

        Uses t-distribution approximation for p-values from r and sample size.
        Applies BH or BY FDR to upper triangle and mirrors.
        """
        mat = self.compute_correlation_matrix(method=method, residualize=residualize)
        if mat is None or mat.empty:
            empty = pd.DataFrame()
            return empty, empty, empty
        users = list(mat.columns)
        # cache
        try:
            frame0 = (
                self.df_resid
                if (residualize and getattr(self, "df_resid", None) is not None)
                else self.df_for_corr
            )
            fp = self._frame_fingerprint(frame0, "resid" if residualize else "for_corr")
            key = f"tgtrax:corr_pq:{khash(method, residualize, fdr_method, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    mat,
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass
        n = len(users)
        # Effective length: drop NaNs pairwise by using notna mask on both columns
        pvals = np.full((n, n), np.nan, dtype=float)
        if student_t is not None:
            iu = np.triu_indices(n, k=1)
            # estimate lag-1 autocorrelation on working frame (for Neff)
            frame0 = (
                self.df_resid
                if residualize and getattr(self, "df_resid", None) is not None
                else self.df_for_corr
            )
            acf1: Dict[int, float] = {}
            if frame0 is not None and not frame0.empty:
                for col_idx, col in enumerate(frame0.columns):
                    s = frame0[col]
                    try:
                        r1 = float(s.autocorr(lag=1))
                    except Exception:
                        r1 = 0.0
                    if np.isnan(r1):
                        r1 = 0.0
                    acf1[col_idx] = max(-0.99, min(0.99, r1))
            for k in range(len(iu[0])):
                i, j = iu[0][k], iu[1][k]
                # Length without NaN
                s_pair = (
                    frame0.iloc[:, [i, j]]
                    if (frame0 is not None and frame0.shape[1] >= j + 1)
                    else None
                )
                if s_pair is None or s_pair.empty:
                    continue
                s2 = s_pair.dropna()
                L = float(len(s2.index))
                if L < 3:
                    continue
                r = float(mat.iat[i, j])
                # Effective sample size (Bartlett lag-1 approx)
                r1x = float(acf1.get(i, 0.0))
                r1y = float(acf1.get(j, 0.0))
                neff = L * (1.0 - r1x * r1y) / (1.0 + r1x * r1y)
                neff = float(max(3.0, min(L, neff)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_stat = r * np.sqrt(max(neff - 2.0, 1.0) / max(1e-12, 1.0 - r * r))
                dfv = max(neff - 2.0, 1.0)
                p = 2.0 * student_t.sf(abs(t_stat), dfv)
                pvals[i, j] = p
                pvals[j, i] = p
            np.fill_diagonal(pvals, np.nan)
        iu = np.triu_indices(n, k=1)
        pvec = pvals[iu]
        if (fdr_method or "bh").lower() == "by":
            qvec = self._by_fdr(pvec)
        else:
            qvec = self._bh_fdr(pvec)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        qvals[iu] = qvec
        qvals = qvals + qvals.T
        np.fill_diagonal(qvals, np.nan)
        out = (
            mat,
            pd.DataFrame(pvals, index=users, columns=users, dtype=float),
            pd.DataFrame(qvals, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(key, {"p": pvals, "q": qvals}, ttl=default_ttl("medium"))
        except Exception:
            pass
        return out

    def get_significant_pairs(
        self, threshold: Optional[float] = None
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identifies pairs of users with correlation above a given threshold.

        Args:
            threshold: The correlation coefficient threshold. If None, the
                       instance\'s default_correlation_threshold is used.
                       The absolute value of the correlation is compared.

        Returns:
            A list of tuples, where each tuple contains:
            ((user1, user2), correlation_value).
            The list is sorted by the absolute correlation value in descending order.
            Returns an empty list if the correlation matrix is empty.
        """
        current_threshold: float = (
            threshold if threshold is not None else self.default_correlation_threshold
        )

        if self.correlation_matrix.empty:
            return []

        significant_pairs: List[Tuple[Tuple[str, str], float]] = []
        columns: List[str] = self.correlation_matrix.columns.tolist()

        # Iterate over the upper triangle of the matrix
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                user1: str = columns[i]
                user2: str = columns[j]
                correlation_value: float = self.correlation_matrix.iloc[i, j]

                if abs(correlation_value) >= current_threshold:
                    significant_pairs.append(((user1, user2), correlation_value))

        significant_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        logger.info(
            f"Found {len(significant_pairs)} significant pairs "
            f"with threshold >= {abs(current_threshold):.2f}"
        )
        return significant_pairs

    def get_crosscorr_significant_pairs(
        self,
        q_threshold: Optional[float] = None,
        min_abs_corr: float = 0.0,
    ) -> List[Tuple[Tuple[str, str], Dict[str, float]]]:
        """Pairs significant by FDR of cross-correlation at best lag.

        Returns list of ((u1,u2), {r, lag_seconds, p, q}). Sorted by |r| desc.
        """
        if self.crosscorr_max.empty or self.crosscorr_qval.empty:
            return []
        qt = self.fdr_alpha if q_threshold is None else q_threshold
        users = list(self.crosscorr_max.columns)
        out: List[Tuple[Tuple[str, str], Dict[str, float]]] = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                r = float(self.crosscorr_max.iat[i, j])
                q = float(self.crosscorr_qval.iat[i, j])
                if np.isnan(q) or q > qt or abs(r) < min_abs_corr:
                    continue
                lag_s = (
                    float(self.crosscorr_best_lag_seconds.iat[i, j])
                    if not self.crosscorr_best_lag_seconds.empty
                    else 0.0
                )
                p = (
                    float(self.crosscorr_pval.iat[i, j])
                    if not self.crosscorr_pval.empty
                    else np.nan
                )
                out.append(
                    (
                        (users[i], users[j]),
                        {"r": r, "lag_seconds": lag_s, "p": p, "q": q},
                    )
                )
        out.sort(key=lambda x: abs(x[1]["r"]), reverse=True)
        logger.info(
            f"Found {len(out)} cross-corr significant pairs (q <= {qt})."
        )
        return out

    def build_correlation_graph(self, threshold: Optional[float] = None) -> nx.Graph:
        """
        Creates a NetworkX graph based on significant user correlations.

        Nodes represent users, and edges represent significant correlations,
        weighted by the correlation coefficient. Community detection (Louvain)
        is applied if edges exist.

        Args:
            threshold: The correlation threshold for including an edge. If None,
                       the instance\'s default_correlation_threshold is used.

        Returns:
            A NetworkX graph. Nodes will have a \'community\' attribute.
        """
        current_threshold: float = (
            threshold if threshold is not None else self.default_correlation_threshold
        )

        graph = nx.Graph()
        if self.df_resampled.empty or not self.user_list:
            logger.warning("Cannot build graph: No resampled data or user list.")
            return graph

        for user in self.user_list:
            graph.add_node(user)

        significant_pairs = self.get_significant_pairs(current_threshold)

        for pair_info in significant_pairs:
            (user1, user2), weight = pair_info
            # Ensure nodes exist (should be guaranteed by prior loop)
            if graph.has_node(user1) and graph.has_node(user2):
                graph.add_edge(user1, user2, weight=round(weight, 3))

        if not graph.edges():
            logger.info(
                "Correlation graph built, but no edges found with threshold %.2f.",
                abs(current_threshold),
            )
        else:
            logger.info(
                "Correlation graph built with %s nodes and %s edges (threshold %.2f).",
                graph.number_of_nodes(),
                graph.number_of_edges(),
                abs(current_threshold),
            )

        # Community detection
        if graph.number_of_edges() > 0:
            try:
                partition: Dict[str, int] = community_louvain.best_partition(
                    graph, weight="weight", random_state=42
                )
                nx.set_node_attributes(graph, partition, "community")
                num_communities: int = len(set(partition.values()))
                logger.info(
                    "Community detection applied. Found %s communities.", num_communities
                )
            except Exception as e:
                logger.error("Error during community detection: %s", e)
        else:
            logger.info(
                "Skipping community detection as there are no edges in the graph."
            )
            default_partition: Dict[str, int] = {node: 0 for node in graph.nodes()}
            nx.set_node_attributes(graph, default_partition, "community")

        return graph

    def build_combined_graph(
        self,
        corr_threshold: Optional[float] = None,
        jaccard_threshold: Optional[float] = None,
        qval_threshold: Optional[float] = None,
        weight_alpha: float = 0.6,  # weight for correlation component
        weight_beta: float = 0.4,  # weight for jaccard component
        residualize: bool = False,
        fdr_method: str = "bh",
        use_runtime_xcorr: bool = True,
        mode: str = "fixed",
    ) -> nx.Graph:
        """Build graph using combined evidence from correlation, cross-correlation, and Jaccard.

        Edge inclusion if any of:
        - |corr| >= corr_threshold
        - qval(best-lag) <= qval_threshold (uses cross-correlation)
        - jaccard >= jaccard_threshold

        Edge weight = weight_alpha * max(|corr|, |crosscorr|) + weight_beta * jaccard.
        """
        ct = (
            self.default_correlation_threshold
            if corr_threshold is None
            else corr_threshold
        )
        jt = (
            self.default_jaccard_threshold
            if jaccard_threshold is None
            else jaccard_threshold
        )
        qt = self.fdr_alpha if qval_threshold is None else qval_threshold

        graph = nx.Graph()
        if not self.user_list:
            return graph
        for u in self.user_list:
            graph.add_node(u)

        users = list(self.user_list)
        # choose matrices
        corr_mat = self.compute_correlation_matrix(
            method=self.corr_method, residualize=residualize
        )
        jacc_mat = self.get_jaccard_matrix()
        if use_runtime_xcorr:
            xr_mat, _lag, _pv, q_mat = self.compute_crosscorr_with_lag(
                method=self.corr_method, residualize=residualize, fdr_method=fdr_method
            )
        else:
            xr_mat, q_mat = self.crosscorr_max, self.crosscorr_qval
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u, v = users[i], users[j]
                corr = (
                    float(corr_mat.get(u, pd.Series()).get(v, np.nan))
                    if not corr_mat.empty
                    else np.nan
                )
                jacc = (
                    float(jacc_mat.get(u, pd.Series()).get(v, np.nan))
                    if not jacc_mat.empty
                    else np.nan
                )
                xr = (
                    float(xr_mat.get(u, pd.Series()).get(v, np.nan))
                    if not xr_mat.empty
                    else np.nan
                )
                qv = (
                    float(q_mat.get(u, pd.Series()).get(v, np.nan))
                    if not q_mat.empty
                    else np.nan
                )

                include = False
                if not np.isnan(corr) and abs(corr) >= ct:
                    include = True
                if not np.isnan(qv) and qv <= qt and not np.isnan(xr):
                    include = True
                if not np.isnan(jacc) and jacc >= jt:
                    include = True

                if include:
                    corr_like = max(
                        abs(corr) if not np.isnan(corr) else 0.0,
                        abs(xr) if not np.isnan(xr) else 0.0,
                    )
                    j_like = jacc if not np.isnan(jacc) else 0.0
                    # fixed vs learned fusion (learned to be plugged via API)
                    if (mode or "fixed").lower() == "learned":
                        # placeholder: fall back to fixed until learned probabilities are provided
                        weight = weight_alpha * corr_like + weight_beta * j_like
                    else:
                        weight = weight_alpha * corr_like + weight_beta * j_like
                    graph.add_edge(
                        u,
                        v,
                        weight=round(weight, 3),
                        corr=round(corr, 3) if not np.isnan(corr) else None,
                        crosscorr=round(xr, 3) if not np.isnan(xr) else None,
                        qval=round(qv, 3) if not np.isnan(qv) else None,
                        jaccard=round(j_like, 3) if j_like else 0.0,
                    )

        # Communities
        if graph.number_of_edges() > 0:
            try:
                partition: Dict[str, int] = community_louvain.best_partition(
                    graph, weight="weight", random_state=42
                )
                nx.set_node_attributes(graph, partition, "community")
                logger.info(
                    "Combined graph: %s nodes, %s edges.",
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                )
                # modularity significance vs degree-preserving null
                try:
                    Q_obs, p_mod = self.modularity_significance(
                        graph, B=200, random_state=42
                    )
                    graph.graph["Q_obs"] = float(Q_obs)
                    graph.graph["p_mod"] = float(p_mod)
                    logger.info("Modularity Q=%.3f, p_mod=%.3f", Q_obs, p_mod)
                except Exception as e:
                    logger.warning("Modularity significance skipped: %s", e)
            except Exception as e:
                logger.error("Community detection failed on combined graph: %s", e)
        return graph

    @staticmethod
    def _part_to_communities(part: Dict[str, int]) -> List[List[str]]:
        comm: Dict[int, List[str]] = defaultdict(list)
        for node, c in part.items():
            comm[int(c)].append(node)
        return list(comm.values())

    def modularity_significance(
        self, G: nx.Graph, B: int = 500, random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """Test observed modularity vs degree-preserving null via rewiring.

        Returns (Q_obs, p_value) where p-value = P(Q_null >= Q_obs).
        """
        try:
            part = community_louvain.best_partition(G, weight="weight", random_state=42)
        except Exception:
            part = {n: 0 for n in G.nodes}
        try:
            Q_obs = nx.algorithms.community.quality.modularity(
                G, self._part_to_communities(part), weight="weight"
            )
        except Exception:
            Q_obs = 0.0
        rng = np.random.default_rng(random_state)
        ge = 0
        m = G.number_of_edges()
        for _ in range(max(1, int(B))):
            H = G.copy()
            try:
                nx.double_edge_swap(H, nswap=max(1, 10 * m), max_tries=100 * m)
            except Exception:
                pass
            try:
                p_b = community_louvain.best_partition(
                    H, weight="weight", random_state=int(rng.integers(0, 1_000_000))
                )
                Q_b = nx.algorithms.community.quality.modularity(
                    H, self._part_to_communities(p_b), weight="weight"
                )
            except Exception:
                Q_b = 0.0
            if Q_b >= Q_obs:
                ge += 1
        p_val = (ge + 1.0) / float(B + 1.0)
        return float(Q_obs), float(p_val)

    def build_te_graph(
        self,
        qval_threshold: Optional[float] = None,
        residualize: bool = True,
        perms: int = 200,
        bootstrap: str = "stationary",
        block_p: float = 0.1,
        fdr_method: str = "by",
    ) -> nx.DiGraph:
        """Build directed graph from Transfer Entropy with q-value threshold.

        Edge i→j exists if q_te[i,j] <= qval_threshold.
        """
        te, pmat, qmat = self.compute_transfer_entropy(
            residualize=residualize,
            perms=perms,
            bootstrap=bootstrap,
            block_p=block_p,
            fdr_method=fdr_method,
        )
        G = nx.DiGraph()
        users = list(te.columns) if not te.empty else self.user_list
        for u in users:
            G.add_node(u)
        qt = float(self.fdr_alpha if qval_threshold is None else qval_threshold)
        for i, u in enumerate(users):
            for j, v in enumerate(users):
                if i == j:
                    continue
                try:
                    qv = float(qmat.iat[i, j])
                    tev = float(te.iat[i, j])
                except Exception:
                    qv, tev = float("nan"), float("nan")
                if not np.isnan(qv) and qv <= qt and not np.isnan(tev):
                    G.add_edge(
                        u,
                        v,
                        te=tev,
                        p_te=(
                            float(pmat.iat[i, j])
                            if (pmat is not None and not pmat.empty)
                            else None
                        ),
                        q_te=qv,
                    )
        return G

    def build_hawkes_graph(
        self,
        qval_threshold: Optional[float] = None,
        half_life_minutes: float = 10.0,
        max_lag_minutes: Optional[float] = None,
        perms: int = 200,
        fdr_method: str = "by",
    ) -> nx.DiGraph:
        """Build directed graph from Hawkes-like excitation with q-value threshold."""
        S, P, Q = self.hawkes_excitation(
            half_life_minutes=half_life_minutes,
            max_lag_minutes=max_lag_minutes,
            perms=perms,
            fdr_method=fdr_method,
        )
        users = list(S.columns) if not S.empty else self.user_list
        G = nx.DiGraph()
        for u in users:
            G.add_node(u)
        qt = float(self.fdr_alpha if qval_threshold is None else qval_threshold)
        for i, u in enumerate(users):
            for j, v in enumerate(users):
                if i == j:
                    continue
                try:
                    qv = float(Q.iat[i, j])
                    sval = float(S.iat[i, j])
                except Exception:
                    qv, sval = float("nan"), float("nan")
                if not np.isnan(qv) and qv <= qt and not np.isnan(sval):
                    G.add_edge(
                        u,
                        v,
                        hawkes=sval,
                        p_hawkes=(
                            float(P.iat[i, j])
                            if (P is not None and not P.empty)
                            else None
                        ),
                        q_hawkes=qv,
                    )
        return G

    # --- Hawkes MLE (pairwise) — placeholder stub for future enhancement ---
    def hawkes_mle_pair(
        self,
        starts_i: np.ndarray,
        starts_j: np.ndarray,
        dt: float,
        max_lag_m: int,
        P: int = 2,
    ) -> Dict[str, Any]:
        """Placeholder for Hawkes MLE on pair (i,j).

        Intended to fit μ, α_ij and kernel parameters via EM/Quasi-Newton and
        return {'alpha_ij': ..., 'llr': ..., 'p_boot': ...}.
        """
        raise NotImplementedError("hawkess_mle_pair not implemented yet")

    def learned_edge_probability(
        self,
        feature_frames_by_window: List[Dict[str, pd.DataFrame]],
        min_repeats: int = 2,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Learn edge probabilities from windowed features (API sketch).

        feature_frames_by_window: list of dicts with keys like {'corr': df, 'jaccard': df, 'xcorr_q': df, 'te': df, ...}
        Returns a symmetric DataFrame of calibrated edge probabilities π_ij.
        """
        # Build labels: edge is 1 if significant in >= min_repeats windows by xcorr_q <= alpha or corr/jaccard thresholds.
        if not feature_frames_by_window:
            return pd.DataFrame()
        users = None
        for d in feature_frames_by_window:
            for k, df in d.items():
                if df is not None and not df.empty:
                    users = list(df.columns)
                    break
            if users:
                break
        if not users:
            return pd.DataFrame()
        n = len(users)
        pair_index: List[Tuple[int, int]] = [
            (i, j) for i in range(n) for j in range(i + 1, n)
        ]
        # Labels
        y = np.zeros(len(pair_index), dtype=int)
        for w, d in enumerate(feature_frames_by_window):
            # decide significance in this window
            sig = np.zeros((n, n), dtype=bool)
            q = d.get("xcorr_q") or d.get("q")
            if q is not None and not q.empty:
                qq = q.to_numpy()
                with np.errstate(invalid="ignore"):
                    sig |= qq <= alpha
            jac = d.get("jaccard")
            if jac is not None and not jac.empty:
                sig |= jac.to_numpy() >= 0.18
            corr = d.get("corr")
            if corr is not None and not corr.empty:
                sig |= np.abs(corr.to_numpy()) >= 0.6
            for idx, (i, j) in enumerate(pair_index):
                if sig[i, j]:
                    y[idx] += 1
        y = (y >= int(min_repeats)).astype(int)
        # Features from latest window if available
        last = feature_frames_by_window[-1]
        feats: List[np.ndarray] = []

        def get_flat(df: Optional[pd.DataFrame], fun=None) -> np.ndarray:
            if df is None or df.empty:
                return np.zeros(len(pair_index), dtype=float)
            A = df.to_numpy()
            if fun is not None:
                A = fun(A)
            return np.array([A[i, j] for (i, j) in pair_index], dtype=float)

        feats.append(get_flat(last.get("corr"), fun=lambda A: np.abs(A)))
        feats.append(get_flat(last.get("jaccard")))
        feats.append(get_flat(last.get("xcorr"), fun=lambda A: np.abs(A)))
        feats.append(
            get_flat(last.get("xcorr_q"), fun=lambda A: 1.0 - np.nan_to_num(A, nan=1.0))
        )
        X = (
            np.stack(feats, axis=1)
            if feats
            else np.zeros((len(pair_index), 1), dtype=float)
        )
        # Simple logistic fit by gradient steps (avoid external deps)
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        w = np.zeros(Xb.shape[1], dtype=float)
        for _ in range(200):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-z))
            g = Xb.T @ (p - y) / Xb.shape[0]
            w -= 0.5 * g  # step size
        p_hat = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        # Build symmetric matrix
        PI = np.full((n, n), np.nan, dtype=float)
        for val, (i, j) in zip(p_hat, pair_index):
            PI[i, j] = PI[j, i] = float(val)
        return pd.DataFrame(PI, index=users, columns=users, dtype=float)

    def get_communities(self, graph: nx.Graph | None = None) -> Dict[int, List[str]]:
        """
        Extracts communities from a graph.

        The graph should have a \'community\' attribute for each node, typically
        set by `build_correlation_graph`.

        Args:
            graph: A NetworkX graph. If None, `build_correlation_graph` is called
                   internally using the instance\'s default threshold.

        Returns:
            A dictionary where keys are community IDs (int) and values are lists
            of usernames (str) in that community. Sorted by community size
            (descending) and then by community ID.
        """
        target_graph: nx.Graph
        if graph is None:
            target_graph = self.build_correlation_graph(
                threshold=self.default_correlation_threshold
            )
        else:
            target_graph = graph

        if not target_graph or not target_graph.nodes():
            logger.warning(
                "Graph is empty or has no nodes. Cannot extract communities."
            )
            return {}

        communities: Dict[int, List[str]] = defaultdict(list)
        for node, data in target_graph.nodes(data=True):
            community_id = data.get("community")
            if community_id is not None:
                communities[community_id].append(node)
            else:
                # Should not happen if graph built by this class
                communities[-1].append(node)  # Unclassified

        # Sort communities by size (desc) then ID (asc for tie-breaking)
        sorted_communities_list = sorted(
            communities.items(),
            key=lambda item: (len(item[1]), item[0]),
            reverse=True,
        )
        return dict(sorted_communities_list)

    def get_users_sorted_by_correlation(
        self,
        top_n: Optional[int] = None,
        centrality: str = "eigenvector",
    ) -> List[str]:
        """
        Sorts users by centrality on the significant correlation graph (preferred),
        falling back to mean absolute correlation.

        Users with no correlation data or only self-correlation will be at the
        end or have a mean correlation of 0.

        Args:
            top_n: If provided, return only the top N users.

        Returns:
            A list of usernames sorted by their mean absolute correlation
            in descending order.
        """
        if self.correlation_matrix.empty or len(self.correlation_matrix.columns) < 2:
            logger.info(
                "Correlation matrix is empty or has less than 2 users. Cannot sort by correlation."
            )
            return self.user_list[:top_n] if top_n else self.user_list
        # Build thresholded graph
        try:
            G = self.build_correlation_graph(
                threshold=self.default_correlation_threshold
            )
            if G.number_of_edges() > 0:
                cent: Dict[str, float]
                if (centrality or "eigenvector").lower().startswith("page"):
                    cent = nx.pagerank(G, weight="weight")
                else:
                    cent = nx.eigenvector_centrality_numpy(G, weight="weight")
                sorted_users = sorted(
                    cent.keys(), key=lambda u: cent.get(u, 0.0), reverse=True
                )
                return sorted_users[:top_n] if top_n else sorted_users
        except Exception:
            pass
        # Fallback: mean absolute correlation
        mean_abs_correlations: Dict[str, float] = {}
        for user in self.correlation_matrix.columns:
            corrs_for_user: pd.Series = self.correlation_matrix[user].drop(user)
            abs_corrs: pd.Series = corrs_for_user.abs()
            mean_abs_correlations[user] = (
                abs_corrs.mean(skipna=True) if abs_corrs.notna().any() else 0.0
            )
        sorted_users = sorted(
            mean_abs_correlations.keys(),
            key=lambda u: mean_abs_correlations[u],
            reverse=True,
        )
        return sorted_users[:top_n] if top_n else sorted_users

    def get_activity_intervals(
        self, user_list: List[str] | None = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates activity intervals (online/offline periods) for users.

        The output format is suitable for `plotly.figure_factory.create_gantt`.

        Args:
            user_list: Optional list of users. If None, uses all users from
                       `self.user_list`.

        Returns:
            A dictionary where keys are usernames and values are lists of dicts,
            each representing an interval:
            {\'Task\': username, \'Start\': start_time_dt,
             \'Finish\': end_time_dt, \'Resource\': \'Online\' or \'Offline\'}
        """
        if self.df_resampled.empty:
            logger.warning(
                "Resampled DataFrame is empty. Cannot generate activity intervals."
            )
            return {}

        target_users: List[str] = user_list if user_list else self.user_list
        if not target_users:
            logger.warning("No target users specified for activity intervals.")
            return {}

        all_user_intervals: Dict[str, List[Dict[str, Any]]] = {}
        # Ensure freq is available, default to 1 minute if not.
        # Use cached frequency seconds
        if self.resample_seconds is None:
            self._infer_resample_offset_and_seconds()
        resample_freq_seconds: float = float(self.resample_seconds or 60.0)

        for user in target_users:
            if user not in self.df_resampled.columns:
                logger.warning(
                    "User %s not found in resampled data. Skipping for interval generation.",
                    user,
                )
                continue

            user_activity: pd.Series = self.df_resampled[user]
            user_intervals: List[Dict[str, Any]] = []

            if user_activity.empty:
                all_user_intervals[user] = []
                continue

            current_status_str: Optional[str] = None
            start_time_dt: Optional[pd.Timestamp] = None

            for i in range(len(user_activity)):
                timestamp: pd.Timestamp = user_activity.index[i]
                status_is_online: bool = user_activity.iloc[i] > 0

                period_status_str: str = "Online" if status_is_online else "Offline"

                if current_status_str is None:
                    current_status_str = period_status_str
                    start_time_dt = timestamp
                elif current_status_str != period_status_str:
                    # Status changed, record previous interval
                    # End time is the start of the current differing one
                    end_time_dt: pd.Timestamp = timestamp
                    if start_time_dt is not None:  # Ensure start_time_dt was set
                        user_intervals.append(
                            {
                                "Task": user,
                                "Start": start_time_dt.to_pydatetime(),
                                "Finish": end_time_dt.to_pydatetime(),
                                "Resource": current_status_str,
                            }
                        )
                    current_status_str = period_status_str
                    start_time_dt = timestamp

            # Record the last interval
            if current_status_str is not None and start_time_dt is not None:
                # End of last interval extends by one resample period duration
                last_interval_end_time: pd.Timestamp = user_activity.index[
                    -1
                ] + pd.to_timedelta(resample_freq_seconds, unit="s")
                user_intervals.append(
                    {
                        "Task": user,
                        "Start": start_time_dt.to_pydatetime(),
                        "Finish": last_interval_end_time.to_pydatetime(),
                        "Resource": current_status_str,
                    }
                )
            all_user_intervals[user] = user_intervals
        return all_user_intervals

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Provides summary statistics about the analyzed data.

        Returns:
            A dictionary containing:
                - num_users (int): Number of users.
                - num_resampled_periods (int): Number of resampled time periods.
                - avg_online_periods_per_user (float): Average number of online
                  periods per user.
                - total_duration_analyzed (str): Human-readable total duration.
        """
        if self.df_resampled.empty:
            return {
                "num_users": 0,
                "num_resampled_periods": 0,
                "avg_online_periods_per_user": 0.0,
                "total_duration_analyzed": "N/A",
            }

        num_users: int = len(self.user_list)
        num_periods: int = len(self.df_resampled)
        avg_online_val: float = self.df_resampled.sum(axis=0).mean()
        avg_online_periods: float = (
            round(avg_online_val, 2) if not pd.isna(avg_online_val) else 0.0
        )

        total_duration_str: str = "N/A"
        # Use cached frequency for calculations
        if num_periods > 0 and self.resample_offset is not None:
            freq_timedelta = pd.to_timedelta(self.resample_offset)
            total_seconds: float
            if num_periods == 1:
                total_seconds = freq_timedelta.total_seconds()
            else:  # num_periods > 1
                total_seconds = (
                    self.df_resampled.index[-1] - self.df_resampled.index[0]
                ).total_seconds() + freq_timedelta.total_seconds()

            days: int = int(total_seconds // (24 * 3600))
            hours: int = int((total_seconds % (24 * 3600)) // 3600)
            minutes: int = int((total_seconds % 3600) // 60)

            freq_str_repr = (
                self.df_resampled.index.freqstr
                if self.df_resampled.index.freqstr
                else "unknown_freq"
            )
            if num_periods == 1:
                total_duration_str = (
                    f"{days}d {hours}h {minutes}m " f"(single period: {freq_str_repr})"
                )
            else:
                total_duration_str = f"{days}d {hours}h {minutes}m"

        elif num_periods > 0:  # Freq unknown but data exists
            min_ts = self.df_resampled.index.min().strftime("%Y-%m-%d %H:%M")
            max_ts = self.df_resampled.index.max().strftime("%Y-%m-%d %H:%M")
            total_duration_str = f"Approx. from {min_ts} to {max_ts} (freq unknown)"

        summary: Dict[str, Any] = {
            "num_users": num_users,
            "num_resampled_periods": num_periods,
            "avg_online_periods_per_user": avg_online_periods,
            "total_duration_analyzed": total_duration_str,
        }
        logger.info("Summary Statistics: %s", summary)
        return summary

    def _calculate_jaccard_indices(self) -> pd.DataFrame:
        """
        Calculates Jaccard index matrix between users' online sessions.

        Returns:
            A pandas DataFrame representing the Jaccard index matrix.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            logger.warning(
                "Not enough data or users to calculate Jaccard indices."
            )
            return pd.DataFrame()

        # Convert to binary (0/1) for Jaccard calculation
        df_binary: pd.DataFrame = self.df_resampled.astype(bool).astype(int)

        users: pd.Index = df_binary.columns

        # Vectorized pairwise intersection using matrix multiplication
        X: np.ndarray = df_binary.to_numpy(dtype=np.int32, copy=False)
        intersections: np.ndarray = X.T @ X  # shape (U, U)

        # Column sums (online counts per user)
        sums: np.ndarray = X.sum(axis=0, dtype=np.int64)  # shape (U,)
        unions: np.ndarray = sums[:, None] + sums[None, :] - intersections

        # Avoid division by zero; where union==0, set jaccard to 0
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard_arr: np.ndarray = np.where(unions > 0, intersections / unions, 0.0)

        # Ensure diagonal is exactly 1.0 when a user has any online activity, else 0.0
        diag_values = np.where(sums > 0, 1.0, 0.0)
        np.fill_diagonal(jaccard_arr, diag_values)

        jaccard_df = pd.DataFrame(jaccard_arr, index=users, columns=users, dtype=float)
        logger.info("Jaccard index matrix calculated (vectorized).")
        return jaccard_df

    # ---- Binary metrics: MCC, Cohen's kappa, Ochiai, Overlap ----
    def _binary_frame(self) -> pd.DataFrame:
        if self.df_resampled.empty:
            return pd.DataFrame()
        return self.df_resampled.astype(bool).astype(int)

    def stability_matrix(
        self,
        method: str = "block",
        residualize: bool = False,
        folds: int = 2,
        min_abs: float = 0.3,
    ) -> pd.DataFrame:
        """Estimate stability of pairwise association across K folds.

        For now, uses Spearman correlation on each fold irrespective of `method`,
        which provides a conservative consistency score:
          stability(u,v) = fraction of folds where |rho_uv| >= min_abs and sign
          matches the majority sign across folds.
        """
        # Block-bootstrap alternative
        if isinstance(method, str) and method.lower() in ("block", "bootstrap", "boot"):
            B = max(50, int(folds) * 50) if folds else 100
            return self.stability_matrix_block(
                residualize=residualize, min_abs=float(min_abs), B=B, p_block=0.1
            )
        folds = max(2, min(6, int(folds)))
        frame = (
            self.df_resid
            if (residualize and getattr(self, "df_resid", None) is not None)
            else self.df_for_corr
        )
        if frame is None or frame.empty or frame.shape[1] < 2:
            return pd.DataFrame()
        users = list(frame.columns)
        T = len(frame.index)
        if T < folds:
            return pd.DataFrame(index=users, columns=users, dtype=float)
        sizes = [T // folds] * folds
        for i in range(T % folds):
            sizes[i] += 1
        # cumulative boundaries
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1] + s)
        mats: List[pd.DataFrame] = []
        for k in range(folds):
            a, b = bounds[k], bounds[k + 1]
            sub = frame.iloc[a:b]
            if sub.shape[0] < 3:
                # degenerate; use zeros
                mats.append(
                    pd.DataFrame(
                        np.zeros((len(users), len(users))), index=users, columns=users
                    )
                )
            else:
                mats.append(sub.corr(method="spearman"))
        # aggregate
        stab = np.zeros((len(users), len(users)), dtype=float)
        sign_votes = np.zeros((len(users), len(users)), dtype=float)
        for m in mats:
            arr = m.to_numpy()
            mask = np.abs(arr) >= float(min_abs)
            stab += mask.astype(float)
            sign_votes += np.sign(arr)
        # majority sign consistency: keep only entries with consistent sign
        sign_consistent = np.sign(sign_votes)
        iu = np.triu_indices(len(users), k=1)
        # normalize by folds
        stab = stab / float(folds)
        stab_df = pd.DataFrame(stab, index=users, columns=users, dtype=float)
        # optional: zero out those with inconsistent sign (mixed signs cancel)
        # here we simply report ratio; filtering is done in API
        return stab_df

    @staticmethod
    def _stationary_bootstrap_indices(
        T: int, p: float = 0.1, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if T <= 1:
            return np.arange(max(T, 0))
        out = np.empty(T, dtype=int)
        i = int(rng.integers(0, T))
        out[0] = i
        for t in range(1, T):
            if rng.random() < p:
                i = int(rng.integers(0, T))
            else:
                i = (i + 1) % T
            out[t] = i
        return out

    def stability_matrix_block(
        self,
        residualize: bool = False,
        min_abs: float = 0.3,
        B: int = 100,
        p_block: float = 0.1,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Block-bootstrap stability using stationary bootstrap indices.

        Returns fraction of resamples where |rho| >= min_abs.
        """
        rng = np.random.default_rng(random_state)
        frame = (
            self.df_resid
            if (residualize and getattr(self, "df_resid", None) is not None)
            else self.df_for_corr
        )
        if frame is None or frame.empty or frame.shape[1] < 2:
            return pd.DataFrame()
        users = list(frame.columns)
        n = len(users)
        T = len(frame.index)
        # cache
        try:
            fp = self._frame_fingerprint(frame, "resid" if residualize else "for_corr")
            key = f"tgtrax:stab_block:{khash(min_abs, B, p_block, residualize, random_state, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return pd.DataFrame(
                    cached["S"], index=users, columns=users, dtype=float
                )
        except Exception:
            pass
        stats = np.zeros((n, n), dtype=float)
        counts = np.zeros((n, n), dtype=int)
        for _ in range(max(1, int(B))):
            idx = self._stationary_bootstrap_indices(T, p=float(p_block), rng=rng)
            sub = frame.iloc[idx]
            try:
                R = sub.corr(method="spearman").to_numpy()
            except Exception:
                R = np.zeros((n, n), dtype=float)
            mask = np.abs(R) >= float(min_abs)
            stats += mask.astype(float)
            counts += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.divide(stats, counts, out=np.zeros_like(stats), where=counts > 0)
        out = pd.DataFrame(S, index=users, columns=users, dtype=float)
        try:
            cache_set(key, {"S": S}, ttl=default_ttl("medium"))
        except Exception:
            pass
        return out

    def jaccard_perm_pvals(
        self,
        perms: int = 100,
        fdr_method: str = "by",
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Permutation p/q-values for Jaccard similarity via circular shifts.

        Shifts one series while holding the other fixed to preserve marginal
        autocorrelation. Returns (pvals, qvals).
        """
        dfb = self._binary_frame()
        if dfb is None or dfb.empty or dfb.shape[1] < 2:
            return pd.DataFrame(), pd.DataFrame()
        users = list(dfb.columns)
        X = dfb.to_numpy(dtype=np.int32)
        T, n = X.shape
        if T < 3:
            return pd.DataFrame(), pd.DataFrame()
        # cache
        try:
            users = list(dfb.columns)
            fp = self._frame_fingerprint(dfb, "binary")
            key = f"tgtrax:jaccard_perm:{khash(perms, fdr_method, random_state, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass

        # observed jaccard
        sums = X.sum(axis=0)
        inter = X.T @ X
        unions = sums[:, None] + sums[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            J_obs = np.where(unions > 0, inter / unions, 0.0)
        rng = np.random.default_rng(random_state)
        ge = np.zeros((n, n), dtype=float)
        for b in range(max(1, int(perms))):
            s = int(rng.integers(0, T))
            Xs = np.roll(X, s, axis=0)
            sums_s = Xs.sum(axis=0)
            inter_s = Xs.T @ X
            unions_s = sums_s[:, None] + sums[None, :] - inter_s
            with np.errstate(divide="ignore", invalid="ignore"):
                J_b = np.where(unions_s > 0, inter_s / unions_s, 0.0)
            ge += (J_b >= J_obs).astype(float)
        pvals = (ge + 1.0) / float(perms + 1)
        np.fill_diagonal(pvals, np.nan)
        iu = np.triu_indices(n, k=1)
        pvec = pvals[iu]
        if (fdr_method or "bh").lower() == "by":
            qvec = self._by_fdr(pvec)
        else:
            qvec = self._bh_fdr(pvec)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        qvals[iu] = qvec
        qvals = qvals + qvals.T
        np.fill_diagonal(qvals, np.nan)
        out = (
            pd.DataFrame(pvals, index=users, columns=users, dtype=float),
            pd.DataFrame(qvals, index=users, columns=users, dtype=float),
        )
        try:
            cache_set(key, {"p": pvals, "q": qvals}, ttl=default_ttl("heavy"))
        except Exception:
            pass
        return out

    def mcc_matrix(self) -> pd.DataFrame:
        dfb = self._binary_frame()
        if dfb.empty or dfb.shape[1] < 2:
            return pd.DataFrame()
        users = list(dfb.columns)
        X = dfb.values.astype(np.float64)
        T = X.shape[0]
        c11 = X.T @ X
        a = X.sum(axis=0)
        b = a.copy()
        c10 = a[:, None] - c11
        c01 = b[None, :] - c11
        c00 = T - a[:, None] - b[None, :] + c11
        num = (c11 * c00) - (c10 * c01)
        denom = a[:, None] * b[None, :] * (T - a)[:, None] * (T - b)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.where(denom <= 0, 0.0, num / np.sqrt(denom))
        # clamp
        M = np.clip(M, -1.0, 1.0)
        return pd.DataFrame(M, index=users, columns=users, dtype=float)

    def kappa_matrix(self) -> pd.DataFrame:
        dfb = self._binary_frame()
        if dfb.empty or dfb.shape[1] < 2:
            return pd.DataFrame()
        users = list(dfb.columns)
        X = dfb.values.astype(np.float64)
        T = X.shape[0]
        c11 = X.T @ X
        a = X.sum(axis=0)
        b = a.copy()
        c10 = a[:, None] - c11
        c01 = b[None, :] - c11
        c00 = T - a[:, None] - b[None, :] + c11
        p_o = (c11 + c00) / float(T)
        p_e = ((a / T)[:, None] * (b / T)[None, :]) + (
            ((T - a) / T)[:, None] * ((T - b) / T)[None, :]
        )
        denom = 1.0 - p_e
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.where(np.abs(denom) < 1e-12, 0.0, (p_o - p_e) / denom)
        return pd.DataFrame(K, index=users, columns=users, dtype=float)

    def ochiai_matrix(self) -> pd.DataFrame:
        dfb = self._binary_frame()
        if dfb.empty or dfb.shape[1] < 2:
            return pd.DataFrame()
        users = list(dfb.columns)
        X = dfb.values.astype(np.float64)
        c11 = X.T @ X
        a = X.sum(axis=0)
        denom = np.sqrt(np.outer(a, a))
        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.where(denom > 0, c11 / denom, 0.0)
        return pd.DataFrame(V, index=users, columns=users, dtype=float)

    def overlap_matrix(self) -> pd.DataFrame:
        dfb = self._binary_frame()
        if dfb.empty or dfb.shape[1] < 2:
            return pd.DataFrame()
        users = list(dfb.columns)
        X = dfb.values.astype(np.float64)
        c11 = X.T @ X
        a = X.sum(axis=0)
        denom = np.minimum(a[:, None], a[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.where(denom > 0, c11 / denom, 0.0)
        return pd.DataFrame(V, index=users, columns=users, dtype=float)

    # ---- Event Synchronization Index (ESI) ----
    def event_sync_index(self, tau_seconds: int = 120) -> pd.DataFrame:
        if self.df_resampled.empty or self.resample_seconds is None:
            return pd.DataFrame()
        users = self.user_list
        dfb = self._binary_frame()
        ev_idx: Dict[str, np.ndarray] = {}
        for u in users:
            s = dfb[u].values
            starts = np.where((s[1:] > s[:-1]))[0] + 1
            ev_idx[u] = starts
        tau_steps = max(0, int(round(tau_seconds / float(self.resample_seconds))))
        n = len(users)
        out = np.full((n, n), np.nan, dtype=float)
        for i, u in enumerate(users):
            A = ev_idx[u]
            for j in range(i, n):
                v = users[j]
                B = ev_idx[v]
                if len(A) == 0 or len(B) == 0:
                    out[i, j] = out[j, i] = 0.0
                    continue
                ai, bi = 0, 0
                matches = 0
                while ai < len(A) and bi < len(B):
                    da = A[ai]
                    db = B[bi]
                    if abs(da - db) <= tau_steps:
                        matches += 1
                        ai += 1
                        bi += 1
                    elif da < db:
                        ai += 1
                    else:
                        bi += 1
                val = matches / float(np.sqrt(max(len(A), 1) * max(len(B), 1)))
                out[i, j] = out[j, i] = float(val)
        return pd.DataFrame(out, index=users, columns=users, dtype=float)

    def event_sync_index_perm_pvals(
        self,
        tau_seconds: int = 120,
        perms: int = 200,
        random_state: Optional[int] = None,
        fdr_method: str = "by",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Permutation p/q-values for ESI via circular shift of one series.

        Returns (pvals, qvals). Diagonals are NaN.
        """
        E = self.event_sync_index(tau_seconds=tau_seconds)
        if E is None or E.empty:
            return pd.DataFrame(), pd.DataFrame()
        users = list(E.columns)
        try:
            fp = self._frame_fingerprint(self._binary_frame(), "binary")
            key = f"tgtrax:esi_perm:{khash(tau_seconds, perms, random_state, fdr_method, tuple(users), fp)}"
            cached = cache_get(key)
            if cached:
                return (
                    pd.DataFrame(cached["p"], index=users, columns=users, dtype=float),
                    pd.DataFrame(cached["q"], index=users, columns=users, dtype=float),
                )
        except Exception:
            pass
        n = len(users)
        dfb = self._binary_frame()
        if dfb is None or dfb.empty:
            return pd.DataFrame(), pd.DataFrame()
        ev_idx = {}
        for u in users:
            s = dfb[u].values
            ev_idx[u] = np.where((s[1:] > s[:-1]))[0] + 1
        tau_steps = max(
            0, int(round(tau_seconds / float(self.resample_seconds or 60.0)))
        )
        rng = np.random.default_rng(random_state)
        P = np.full((n, n), np.nan, dtype=float)
        T = len(self.df_resampled.index)
        for i in range(n):
            for j in range(i + 1, n):
                A = ev_idx[users[i]]
                B = ev_idx[users[j]]
                if len(A) == 0 or len(B) == 0:
                    continue
                # observed ESI
                ai = bi = 0
                matches = 0
                while ai < len(A) and bi < len(B):
                    da = A[ai]
                    db = B[bi]
                    if abs(da - db) <= tau_steps:
                        matches += 1
                        ai += 1
                        bi += 1
                    elif da < db:
                        ai += 1
                    else:
                        bi += 1
                obs = matches / float(np.sqrt(max(len(A), 1) * max(len(B), 1)))
                ge = 0
                for b in range(max(1, int(perms))):
                    sft = int(rng.integers(0, T))
                    B_s = (B + sft) % T
                    B_s.sort()
                    ai = bi = 0
                    matches_b = 0
                    while ai < len(A) and bi < len(B_s):
                        da = A[ai]
                        db = B_s[bi]
                        if abs(da - db) <= tau_steps:
                            matches_b += 1
                            ai += 1
                            bi += 1
                        elif da < db:
                            ai += 1
                        else:
                            bi += 1
                    val = matches_b / float(np.sqrt(max(len(A), 1) * max(len(B_s), 1)))
                    if val >= obs:
                        ge += 1
                p = (ge + 1.0) / float(perms + 1.0)
                P[i, j] = P[j, i] = p
        iu = np.triu_indices(n, k=1)
        pvec = P[iu]
        qvec = (
            self._by_fdr(pvec)
            if (fdr_method or "by").lower() == "by"
            else self._bh_fdr(pvec)
        )
        Q = np.full_like(P, np.nan, dtype=float)
        Q[iu] = qvec
        Q = Q + Q.T
        np.fill_diagonal(P, np.nan)
        np.fill_diagonal(Q, np.nan)
        outp = pd.DataFrame(P, index=users, columns=users, dtype=float)
        outq = pd.DataFrame(Q, index=users, columns=users, dtype=float)
        try:
            cache_set(key, {"p": P, "q": Q}, ttl=default_ttl("heavy"))
        except Exception:
            pass
        return (outp, outq)

    def get_jaccard_matrix(self) -> pd.DataFrame:
        """
        Retrieves the calculated Jaccard index matrix.

        Returns:
            A pandas DataFrame with user-to-user Jaccard indices.
        """
        return self.jaccard_matrix

    def get_significant_jaccard_pairs(
        self, threshold: Optional[float] = None
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identifies pairs of users with Jaccard index above a given threshold.

        Args:
            threshold: The Jaccard index threshold. If None, the
                       instance's default_jaccard_threshold is used.

        Returns:
            A list of tuples, where each tuple contains:
            ((user1, user2), jaccard_index).
            The list is sorted by the Jaccard index in descending order.
        """
        current_threshold: float = (
            threshold if threshold is not None else self.default_jaccard_threshold
        )

        if self.jaccard_matrix.empty:
            return []

        significant_pairs: List[Tuple[Tuple[str, str], float]] = []
        columns: List[str] = self.jaccard_matrix.columns.tolist()

        # Iterate over the upper triangle of the matrix
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                user1: str = columns[i]
                user2: str = columns[j]
                jaccard_value: float = self.jaccard_matrix.iloc[i, j]

                if jaccard_value >= current_threshold:
                    significant_pairs.append(((user1, user2), jaccard_value))

        significant_pairs.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "Found %s significant Jaccard pairs with threshold >= %.2f",
            len(significant_pairs),
            current_threshold,
        )
        return significant_pairs


# --- Helper Functions ---


def create_activity_gantt_chart(
    activity_intervals: Dict[str, List[Dict[str, Any]]],
    title: str = "User Activity Gantt Chart",
) -> Union[Any, None]:  # Plotly figure type is not standard, use Any
    """
    Generates a Plotly Gantt chart from activity intervals.

    Requires Plotly to be installed.

    Args:
        activity_intervals: A dictionary of activity intervals, typically from
                            `TemporalAnalyzer.get_activity_intervals()`.
        title: The title of the Gantt chart.

    Returns:
        A Plotly Figure object if successful, otherwise None.
    """
    try:
        import plotly.figure_factory as ff
        import plotly.io as pio  # For potential saving, not used directly here for display

        # pio.templates.default = "plotly_dark" # Optional: theme
    except ImportError:
        logger.warning(
            "Plotly is not installed. Cannot generate Gantt chart. Please install with `pip install plotly`."
        )
        return None

    df_tasks: List[Dict[str, Any]] = []
    for user_intervals in activity_intervals.values():
        df_tasks.extend(user_intervals)

    if not df_tasks:
        logger.info("No data available to generate Gantt chart.")
        return None

    colors: Dict[str, str] = {
        "Online": "rgb(0, 200, 0)",  # Green
        "Offline": "rgb(220, 0, 0)",  # Red
    }

    try:
        fig = ff.create_gantt(
            df_tasks,
            colors=colors,
            index_col="Resource",  # \'Online\' or \'Offline\'
            show_colorbar=True,
            group_tasks=True,  # Groups by \'Task\' (username)
            title=title,
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="User")
        logger.info(
            "Gantt chart '%s' created. Display with fig.show() or save externally.",
            title,
        )
        # Example save:
        # chart_filename = title.lower().replace(' ', '_') + ".html"
        # pio.write_html(fig, file=chart_filename, auto_open=False)
        # tui.tui_print_info(f"Gantt chart saved to {chart_filename}")
        return fig
    except Exception as e:
        logger.error("Error creating Gantt chart: %s", e)
        return None


# --- Main Execution Block (for testing and demonstration) ---

if __name__ == "__main__":
    logger.info("Testing TemporalAnalyzer module...")

    # Dummy data setup
    date_rng = pd.date_range(
        start="2023-01-01 00:00",
        end="2023-01-01 01:00",
        freq="1min",
        tz="UTC",
    )
    data_payload = {
        "Alice": [
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
        ],
        "Bob": [
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        ],
        "Charlie": [
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
        ],
    }
    df_activity_test = pd.DataFrame(data_payload, index=date_rng).astype(bool)

    logger.info("Initializing TemporalAnalyzer with dummy data...")
    analyzer = TemporalAnalyzer(
        df_activity_test, resample_period="2T", correlation_threshold=0.3
    )

    logger.info("\\n--- Correlation Matrix ---")
    corr_matrix_test = analyzer.get_correlation_matrix()
    if not corr_matrix_test.empty:
        logger.debug(corr_matrix_test.to_string())
    else:
        logger.warning("Correlation matrix is empty.")

    logger.info("\\n--- Significant Pairs (threshold from init) ---")
    sig_pairs_default_test = analyzer.get_significant_pairs()
    if sig_pairs_default_test:
        for pair, corr_val in sig_pairs_default_test:
            logger.info(f"  {pair}: {corr_val:.3f}")
    else:
        logger.warning("No significant pairs found with default threshold.")

    logger.info("\\n--- Significant Pairs (threshold override 0.1) ---")
    sig_pairs_override_test = analyzer.get_significant_pairs(threshold=0.1)
    if sig_pairs_override_test:
        for pair, corr_val in sig_pairs_override_test:
            logger.info(f"  {pair}: {corr_val:.3f}")
    else:
        logger.warning(
            "No significant pairs found with overridden threshold 0.1."
        )

    logger.info("\\n--- Building Correlation Graph (threshold from init) ---")
    graph_test = analyzer.build_correlation_graph()
    if graph_test.nodes():
        logger.info(
            f"  Graph nodes: {graph_test.number_of_nodes()}, "
            f"edges: {graph_test.number_of_edges()}"
        )
        if graph_test.edges():
            logger.info("  Communities:")
            communities_test = analyzer.get_communities(graph=graph_test)
            for comm_id, users_in_comm in communities_test.items():
                logger.info(f"    Community {comm_id}: {users_in_comm}")
    else:
        logger.warning("Graph is empty.")

    logger.info("\\n--- Users Sorted by Correlation ---")
    sorted_users_test = analyzer.get_users_sorted_by_correlation(top_n=5)
    logger.info(f"  Top 5 correlated users: {sorted_users_test}")

    logger.info("\\n--- Activity Intervals (for Gantt) ---")
    activity_intervals_data_test = analyzer.get_activity_intervals()
    if (
        "Alice" in activity_intervals_data_test
        and activity_intervals_data_test["Alice"]
    ):
        logger.info("Sample intervals for Alice:")
        for interval in activity_intervals_data_test["Alice"][:3]:
            start_str = interval["Start"].strftime("%Y-%m-%d %H:%M:%S")
            finish_str = interval["Finish"].strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"  Start: {start_str}, Finish: {finish_str}, "
                f"Resource: {interval['Resource']}"
            )
        # gantt_fig_test = create_activity_gantt_chart(
        #     activity_intervals_data_test, title="Test User Activity"
        # )
        # if gantt_fig_test:
        #     pass # fig.show() or save
    else:
        logger.warning("No activity intervals found for Alice or data is empty.")

    logger.info("\\n--- Summary Stats ---")
    summary_test = analyzer.get_summary_stats()
    for key, value in summary_test.items():
        logger.info(f"  {key.replace('_',' ').capitalize()}: {value}")

    logger.info("\\nTemporalAnalyzer tests completed.")
