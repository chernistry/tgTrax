import useSWR from 'swr'
import Navbar from '../components/Navbar'
import { apiBase, fetchJSON } from '../lib/api'
import { useState } from 'react'
import { useRouter } from 'next/router'
import dynamic from 'next/dynamic'
import { Button, Card, Alert } from 'antd'
import { useI18n } from '../lib/i18n'

// AG Grid (dynamic to avoid SSR issues)
const AgGridReact = dynamic(async () => (await import('ag-grid-react')).AgGridReact, { ssr: false }) as any
import 'ag-grid-community/styles/ag-grid.css'
import 'ag-grid-community/styles/ag-theme-alpine.css'

type Pair = { u: string, v: string, score?: number|null, r?: number|null, lag_seconds?: number|null, p?: number|null, q?: number|null, jaccard?: number|null, corr?: number|null, source?: string, L?: number|null, nnz_u?: number|null, nnz_v?: number|null, coincidences?: number|null, stability?: number|null }

type Resp = { pairs: Pair[], meta?: Record<string, any> }

export default function Pairs() {
  const router = useRouter()
  const since = (router.query?.since as string) || 'start'
  const [method, setMethod] = useState<'xcorr'|'spearman'|'pearson'|'mcc'|'kappa'|'ochiai'|'overlap'|'esi'|'te'|'hawkes'>('xcorr')
  const [resid, setResid] = useState(false)
  const [permute, setPermute] = useState(false)
  const [perms, setPerms] = useState(100)
  const [tau, setTau] = useState(120)
  const [halfLife, setHalfLife] = useState(10)
  const [maxLag, setMaxLag] = useState(30)
  const [fdr, setFdr] = useState<'bh'|'by'>('by')
  const [bootstrap, setBootstrap] = useState<'stationary'|'shift'>('stationary')
  const [blockP, setBlockP] = useState(0.1)
  const [q, setQ] = useState(0.05)
  const [minAbs, setMinAbs] = useState(0.3)
  const [top, setTop] = useState(200)
  const [minL, setMinL] = useState(0)
  const [minNonzero, setMinNonzero] = useState(0)
  const [stable, setStable] = useState(false)
  const [folds, setFolds] = useState(3)
  const [stabilityMin, setStabilityMin] = useState(0.67)
  const [period, setPeriod] = useState<'auto'|'1min'|'3min'|'5min'>('auto')
  const [agg, setAgg] = useState<'max'|'mean'|'sum'|'any'>('max')
  const [fill, setFill] = useState<'zero'|'ffill'|'nan'>('zero')
  const [teQuantize, setTeQuantize] = useState<'balanced'|'median'|'global_median'|'global_mean'|'zero'>('balanced')
  const { t } = useI18n()

  const baseParams: Record<string, string> = { method, residualize: String(resid), q_threshold: String(q), min_abs_corr: String(minAbs), top: String(top), since, fdr_method: fdr, period, resample_agg: agg, fill_missing: fill }
  if (method === 'xcorr') {
    baseParams['permute'] = String(permute)
    baseParams['perms'] = String(perms)
    baseParams['tau_seconds'] = String(tau)
  }
  if (method === 'esi') {
    baseParams['tau_seconds'] = String(tau)
  }
  if (method === 'te') {
    baseParams['perms'] = String(perms)
    baseParams['bootstrap'] = bootstrap
    baseParams['block_p'] = String(blockP)
    baseParams['te_quantize'] = teQuantize
  }
  if (method === 'hawkes') {
    baseParams['perms'] = String(perms)
    baseParams['half_life_minutes'] = String(halfLife)
    baseParams['max_lag_minutes'] = String(maxLag)
  }
  baseParams['min_L'] = String(minL)
  baseParams['min_nonzero'] = String(minNonzero)
  baseParams['stable'] = String(stable)
  baseParams['folds'] = String(folds)
  baseParams['stability_min'] = String(stabilityMin)
  if (method === 'xcorr' || method === 'esi') {
    baseParams['tau_seconds'] = String(tau)
  }
  const params = new URLSearchParams(baseParams)
  const url = `${apiBase()}/api/pairs/significant?${params.toString()}`
  const { data, error, isLoading } = useSWR<Resp>(url, fetchJSON)

  function classify(p: Pair): 'good'|'warn'|'bad' {
    const m = p.metrics || {}
    const qv = (m.q ?? null) as number|null
    const xcorr = (m.xcorr ?? null) as number|null
    const spearman = (m.spearman ?? null) as number|null
    const jacc = (m.jaccard ?? null) as number|null
    const esi = (m.esi ?? null) as number|null
    const teq = (m.te_q ?? null) as number|null
    const te = (m.te ?? null) as number|null
    const agree = p.agree_count || 0

    const abs = (v: number|null) => v == null ? 0 : Math.abs(v)
    const strongXCorr = abs(xcorr) >= 0.4
    const modXCorr = abs(xcorr) >= 0.3
    const modSpearman = abs(spearman) >= 0.3
    const modJ = (jacc ?? 0) >= 0.2
    const modESI = (esi ?? 0) >= 0.45
    const qOK = (qv != null && qv <= 0.05)
    const teOK = (teq != null && teq <= 0.05)

    if (teOK || (te != null && te > 0.05)) return 'good'
    if (qOK && strongXCorr) return 'good'
    if (qOK && modXCorr && modSpearman) return 'good'
    if (agree >= 4 && (strongXCorr || abs(spearman) >= 0.4)) return 'good'
    if (modXCorr || modSpearman || modJ || modESI || agree >= 2) return 'warn'
    return 'bad'
  }

  function verdictLabel(cls: 'good'|'warn'|'bad') {
    if (cls === 'good') return <span className="pill ok">{t('pairs.verdict.good')}</span>
    if (cls === 'warn') return <span className="pill warn">{t('pairs.verdict.warn')}</span>
    return <span className="pill err">{t('pairs.verdict.bad')}</span>
  }

  return (
    <>
      <Navbar />
      <div className="container">
        <h1 className="title">{t('pairs.title')}</h1>
        <Alert type="info" showIcon message={t('pairs.alert')} style={{marginBottom:12}} />
        <Card size="small" style={{marginBottom:16}} title={t('pairs.filters')}>
          <div style={{display:'flex', gap:12, alignItems:'center', marginTop:8, flexWrap:'wrap'}}>
            <Button type="primary" onClick={()=>{
              setMethod('xcorr')
              setResid(true)
              setPermute(true)
              setPerms(200)
              setFdr('by')
              setQ(0.05)
              setMinAbs(0.3)
              setTop(200)
              setMinL(100)
              setMinNonzero(10)
              setStable(true)
              setFolds(3)
              setStabilityMin(0.67)
              setPeriod('auto')
              setAgg('max')
              setFill('zero')
              setTau(120)
              setBootstrap('stationary')
              setBlockP(0.1)
              setTeQuantize('balanced')
            }}>{t('pairs.makeAwesome')}</Button>
            <label>{t('pairs.method')}
              <select value={method} onChange={e=>setMethod(e.target.value as any)} style={{marginLeft:6}}>
                <option value="xcorr">{t('pairs.method.xcorr')}</option>
                <option value="spearman">{t('pairs.method.spearman')}</option>
                <option value="pearson">{t('pairs.method.pearson')}</option>
                <option value="mcc">{t('pairs.method.mcc')}</option>
                <option value="kappa">{t('pairs.method.kappa')}</option>
                <option value="ochiai">{t('pairs.method.ochiai')}</option>
                <option value="overlap">{t('pairs.method.overlap')}</option>
                <option value="esi">{t('pairs.method.esi')}</option>
                <option value="te">{t('pairs.method.te')}</option>
                <option value="hawkes">{t('pairs.method.hawkes')}</option>
              </select>
            </label>
            <label><input type="checkbox" checked={resid} onChange={e=>setResid(e.target.checked)} /> {t('pairs.residualize')}</label>
            {method === 'xcorr' && (
              <label><input type="checkbox" checked={permute} onChange={e=>setPermute(e.target.checked)} /> {t('pairs.permute')}</label>
            )}
            {(method === 'xcorr' || method === 'te' || method === 'hawkes') && (
              <label>{t('pairs.perms')} <input type="number" min={50} max={500} value={perms} onChange={e=>setPerms(parseInt(e.target.value || '0', 10))} style={{width:80, marginLeft:6}}/></label>
            )}
            {(method === 'xcorr' || method === 'esi') && (
              <label>{t('pairs.tau')} <input type="number" min={10} step={10} value={tau} onChange={e=>setTau(parseInt(e.target.value || '0', 10))} style={{width:100, marginLeft:6}}/></label>
            )}
            <label>{t('pairs.fdr')}
              <select value={fdr} onChange={e=>setFdr(e.target.value as any)} style={{marginLeft:6}}>
                <option value="bh">BH</option>
                <option value="by">BY</option>
              </select>
            </label>
            {method === 'te' && (
              <>
                <label>{t('pairs.bootstrap')}
                  <select value={bootstrap} onChange={e=>setBootstrap(e.target.value as any)} style={{marginLeft:6}}>
                    <option value="stationary">stationary</option>
                    <option value="shift">shift</option>
                  </select>
                </label>
                <label>{t('pairs.blockP')} <input type="number" step={0.01} min={0.01} max={0.99} value={blockP} onChange={e=>setBlockP(parseFloat(e.target.value || '0'))} style={{width:100, marginLeft:6}}/></label>
                <label>{t('pairs.teQuantize')}
                  <select value={teQuantize} onChange={e=>setTeQuantize(e.target.value as any)} style={{marginLeft:6}}>
                    <option value="balanced">balanced</option>
                    <option value="median">median</option>
                    <option value="global_median">global median</option>
                    <option value="global_mean">global mean</option>
                    <option value="zero">zero</option>
                  </select>
                </label>
              </>
            )}
            {method === 'hawkes' && (
              <>
                <label>{t('pairs.halfLife')} <input type="number" min={1} value={halfLife} onChange={e=>setHalfLife(parseFloat(e.target.value || '0'))} style={{width:120, marginLeft:6}}/></label>
                <label>{t('pairs.maxLag')} <input type="number" min={1} value={maxLag} onChange={e=>setMaxLag(parseFloat(e.target.value || '0'))} style={{width:120, marginLeft:6}}/></label>
              </>
            )}
            <label>{t('pairs.qThreshold')} <input type="number" step={0.01} value={q} onChange={e=>setQ(parseFloat(e.target.value || '0'))} style={{width:100, marginLeft:6}}/></label>
            <label>{t('pairs.minAbs')} <input type="number" step={0.01} value={minAbs} onChange={e=>setMinAbs(parseFloat(e.target.value || '0'))} style={{width:100, marginLeft:6}}/></label>
            <label>{t('pairs.topN')} <input type="number" step={10} value={top} onChange={e=>setTop(parseInt(e.target.value || '0', 10))} style={{width:100, marginLeft:6}}/></label>
            <label>{t('pairs.minL')} <input type="number" min={0} value={minL} onChange={e=>setMinL(parseInt(e.target.value || '0', 10))} style={{width:80, marginLeft:6}}/></label>
            <label>{t('pairs.minNonzero')} <input type="number" min={0} value={minNonzero} onChange={e=>setMinNonzero(parseInt(e.target.value || '0', 10))} style={{width:110, marginLeft:6}}/></label>
            <label><input type="checkbox" checked={stable} onChange={e=>setStable(e.target.checked)} /> {t('pairs.stability')}</label>
            {stable && (
              <>
                <label>{t('pairs.folds')} <input type="number" min={2} max={6} value={folds} onChange={e=>setFolds(parseInt(e.target.value || '3', 10))} style={{width:80, marginLeft:6}}/></label>
                <label>{t('pairs.stabilityMin')} <input type="number" step={0.01} min={0} max={1} value={stabilityMin} onChange={e=>setStabilityMin(parseFloat(e.target.value || '0.67'))} style={{width:90, marginLeft:6}}/></label>
              </>
            )}
            <label>{t('pairs.period')}
              <select value={period} onChange={e=>setPeriod(e.target.value as any)} style={{marginLeft:6}}>
                <option value="auto">auto</option>
                <option value="1min">1min</option>
                <option value="3min">3min</option>
                <option value="5min">5min</option>
              </select>
            </label>
            <label>{t('pairs.agg')}
              <select value={agg} onChange={e=>setAgg(e.target.value as any)} style={{marginLeft:6}}>
                <option value="max">max</option>
                <option value="mean">mean</option>
                <option value="sum">sum</option>
                <option value="any">any</option>
              </select>
            </label>
            <label>{t('pairs.fill')}
              <select value={fill} onChange={e=>setFill(e.target.value as any)} style={{marginLeft:6}}>
                <option value="zero">zero</option>
                <option value="ffill">ffill</option>
                <option value="nan">nan</option>
              </select>
            </label>
          </div>
        </Card>
        {isLoading && <Alert type="info" message={t('common.loading')} />}
        {error && <Alert type="error" message={t('common.loadError')} />}
        {data?.meta?.nbs_p != null && (
          <Alert
            style={{ marginBottom: 12 }}
            type={data.meta.nbs_p <= 0.05 ? 'success' : 'warning'}
            showIcon
            message={
              <span>
                {t('pairs.nbs', { value: data.meta.nbs_p.toFixed(3) })}
                {data.meta.efp_upper != null && (
                  <>
                    {' · '}
                    {t('pairs.efp', { value: Number(data.meta.efp_upper).toFixed(1) })}
                  </>
                )}
              </span>
            }
          />
        )}
        {data && (
          <Card>
            {data.pairs.length === 0 && (
              <Alert type="warning" message={t('pairs.empty')} />
            )}
            {data.pairs.length > 0 && (
              <div className="ag-theme-alpine" style={{ width: '100%', height: 600, background: 'transparent' }}>
                <AgGridReact
                  rowData={data.pairs.map((p, i) => ({ idx: i + 1, ...p }))}
                  columnDefs={[
                    { headerName: t('pairs.col.idx'), field: 'idx', width: 70, sortable: true },
                    { headerName: t('pairs.col.userA'), field: 'u', sortable: true, filter: true },
                    { headerName: t('pairs.col.userB'), field: 'v', sortable: true, filter: true },
                    { headerName: t('pairs.col.score'), field: 'score', sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(3)) },
                    { headerName: t('pairs.col.r'), field: 'r', sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(3)) },
                    { headerName: t('pairs.col.lag'), field: 'lag_seconds', sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(0)) },
                    { headerName: t('pairs.col.q'), field: 'q', sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(3)) },
                    { headerName: t('pairs.col.jacc'), field: 'jaccard', sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(3)) },
                    { headerName: t('pairs.col.corr'), field: 'corr', sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(3)) },
                    { headerName: t('pairs.col.L'), field: 'L', width: 80, sortable: true },
                    { headerName: t('pairs.col.nnzU'), field: 'nnz_u', width: 100, sortable: true },
                    { headerName: t('pairs.col.nnzV'), field: 'nnz_v', width: 100, sortable: true },
                    { headerName: t('pairs.col.coinc'), field: 'coincidences', width: 90, sortable: true },
                    { headerName: t('pairs.col.stab'), field: 'stability', width: 90, sortable: true, valueFormatter: (p:any) => (p.value == null ? '-' : Number(p.value).toFixed(2)) },
                    { headerName: t('pairs.col.source'), field: 'source', sortable: true },
                  ]}
                  defaultColDef={{ resizable: true, sortable: true, filter: true }}
                  animateRows={true}
                  pagination={true}
                  paginationPageSize={50}
                />
              </div>
            )}
            <div className="row" style={{marginTop:16, display:'grid', gap:16}}>
              {data.pairs.map((p, idx) => {
                const cls = classify(p)
                return (
                  <div className="card" key={`${p.u}-${p.v}-${idx}`} style={{flex:1, border:'1px solid #2b3e61'}}>
                    <div className="title">{t('pairs.card.title', { rank: idx + 1, a: p.u, b: p.v })}</div>
                    <div style={{display:'flex', gap:8, flexWrap:'wrap', marginBottom:8}}>
                      <span className="pill ok">{t('pairs.badge.matches', { count: p.agree_count ?? 0 })}</span>
                      <span className="pill">{t('pairs.badge.score', { score: p.score?.toFixed(3) ?? '—' })}</span>
                      {typeof p.lag_seconds === 'number' && <span className="pill">{t('pairs.badge.lag', { lag: Math.round(p.lag_seconds) })}</span>}
                      {p.leader && <span className="pill warn">{t('pairs.badge.leader', { leader: p.leader })}</span>}
                      {verdictLabel(cls)}
                    </div>
                    {p.reason && <div className="muted" style={{marginBottom:8}}>{t('pairs.reason', { reason: p.reason })}</div>}
                    <div className="muted">{t('pairs.metrics')}</div>
                    <ul style={{marginTop:6, lineHeight:1.6}}>
                      <li>{t('pairs.metric.q')}: {p.metrics?.q == null ? '-' : p.metrics.q.toFixed(3)}</li>
                      <li>{t('pairs.metric.absXcorr')}: {p.metrics?.xcorr == null ? '-' : Math.abs(p.metrics.xcorr).toFixed(3)}</li>
                      <li>{t('pairs.metric.absSpearman')}: {p.metrics?.spearman == null ? '-' : Math.abs(p.metrics.spearman).toFixed(3)}</li>
                      <li>{t('pairs.metric.jaccard')}: {p.metrics?.jaccard == null ? '-' : p.metrics.jaccard.toFixed(3)}</li>
                      <li>{t('pairs.metric.jaccardQ')}: {p.metrics?.jaccard_q == null ? '-' : p.metrics.jaccard_q.toFixed(3)}</li>
                      <li>{t('pairs.metric.mcc')}: {p.metrics?.mcc == null ? '-' : p.metrics.mcc.toFixed(3)}</li>
                      <li>{t('pairs.metric.kappa')}: {p.metrics?.kappa == null ? '-' : p.metrics.kappa.toFixed(3)}</li>
                      <li>{t('pairs.metric.ochiai')}: {p.metrics?.ochiai == null ? '-' : p.metrics.ochiai.toFixed(3)}</li>
                      <li>{t('pairs.metric.overlap')}: {p.metrics?.overlap == null ? '-' : p.metrics.overlap.toFixed(3)}</li>
                      <li>{t('pairs.metric.esi')}: {p.metrics?.esi == null ? '-' : p.metrics.esi.toFixed(3)}</li>
                      <li>{t('pairs.metric.te')}: {p.metrics?.te == null ? '-' : p.metrics.te.toFixed(3)}; {t('pairs.metric.teq')}: {p.metrics?.te_q == null ? '-' : p.metrics.te_q.toFixed(3)}</li>
                    </ul>
                  </div>
                )
              })}
            </div>
          </Card>
        )}
      </div>
    </>
  )
}
