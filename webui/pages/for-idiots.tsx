import useSWR from 'swr'
import Navbar from '../components/Navbar'
import { Card, Button, Alert, Tag } from 'antd'
import { apiBase, fetchJSON } from '../lib/api'
import { useState, useMemo } from 'react'
import { useRouter } from 'next/router'
import { useI18n } from '../lib/i18n'

type Pair = {
  u: string
  v: string
  score: number
  agree_count: number
  leader?: string | null
  lag_seconds?: number | null
  metrics: Record<string, number | null>
  reason: string
}
type Resp = { pairs: Pair[], meta: Record<string, any> }

export default function ForDummies() {
  const router = useRouter()
  const since = (router.query?.since as string) || 'start'
  const [resid, setResid] = useState(false)
  const [permute, setPermute] = useState(false)
  const [perms, setPerms] = useState(100)
  const [tau, setTau] = useState(120)
  const [fdr, setFdr] = useState<'bh'|'by'>('bh')
  const [useTe, setUseTe] = useState(false)
  const [permJ, setPermJ] = useState(false)
  const { t } = useI18n()

  const params = new URLSearchParams({
    since,
    limit: '3',
    residualize: String(resid),
    permute: String(permute),
    perms: String(perms),
    tau_seconds: String(tau),
    fdr_method: fdr,
    use_te: String(useTe),
    permute_jaccard: String(permJ),
  })
  const url = `${apiBase()}/api/consensus/top?${params.toString()}`
  const { data, error, isLoading } = useSWR<Resp>(url, fetchJSON)

  const cards = useMemo(() => data?.pairs ?? [], [data])

  function classify(p: Pair): 'good'|'warn'|'bad' {
    const m = p.metrics || {}
    const q = (m.q ?? null) as number | null
    const xcorr = (m.xcorr ?? null) as number | null
    const spearman = (m.spearman ?? null) as number | null
    const jq = (m.jaccard_q ?? null) as number | null
    const jacc = (m.jaccard ?? null) as number | null
    const esi = (m.esi ?? null) as number | null
    const teq = (m.te_q ?? null) as number | null
    const te = (m.te ?? null) as number | null
    const agree = p.agree_count || 0

    const abs = (v: number | null) => (v == null ? 0 : Math.abs(v))
    const strongXCorr = abs(xcorr) >= 0.4
    const modXCorr = abs(xcorr) >= 0.3
    const modSpearman = abs(spearman) >= 0.3
    const modJ = (jacc ?? 0) >= 0.2
    const modESI = (esi ?? 0) >= 0.45
    const qOK = q != null && q <= 0.05
    const jqOK = jq != null && jq <= 0.05
    const teOK = teq != null && teq <= 0.05

    if (teOK || (te != null && te > 0.05)) return 'good'
    if (qOK && strongXCorr) return 'good'
    if (jqOK && (modXCorr || modSpearman)) return 'good'
    if (agree >= 4 && (strongXCorr || abs(spearman) >= 0.4)) return 'good'
    if (modXCorr || modSpearman || modJ || modESI || agree >= 2) return 'warn'
    return 'bad'
  }

  function cardStyle(cls: 'good'|'warn'|'bad'): React.CSSProperties {
    if (cls === 'good') return { border: '1px solid #2ecc71', boxShadow: '0 0 0 2px rgba(46,204,113,0.18) inset' }
    if (cls === 'warn') return { border: '1px solid #f1c40f', boxShadow: '0 0 0 2px rgba(241,196,15,0.12) inset' }
    return { border: '1px solid #e74c3c', boxShadow: '0 0 0 2px rgba(231,76,60,0.12) inset' }
  }

  function verdictLabel(cls: 'good'|'warn'|'bad') {
    if (cls === 'good') return <span className="pill ok">{t('dumb.verdict.good')}</span>
    if (cls === 'warn') return <span className="pill warn">{t('dumb.verdict.warn')}</span>
    return <span className="pill err">{t('dumb.verdict.bad')}</span>
  }

  return (
    <>
      <Navbar />
      <div className="container">
        <h1 className="title">{t('dumb.title')}</h1>
        <p className="muted">{t('dumb.subtitle')}</p>
        <Card size="small" style={{ marginBottom: 16 }}>
          <div className="muted">{t('dumb.params')}</div>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 8, flexWrap: 'wrap' }}>
            <Button type="primary" onClick={() => { setResid(true); setPermute(true); setPerms(200); setTau(120); setFdr('by'); setUseTe(true); setPermJ(true) }}>
              {t('dumb.makeAwesome')}
            </Button>
            <label><input type="checkbox" checked={resid} onChange={e => setResid(e.target.checked)} /> {t('dumb.residualize')}</label>
            <label><input type="checkbox" checked={permute} onChange={e => setPermute(e.target.checked)} /> {t('dumb.permute')}</label>
            <label>{t('dumb.perms')} <input type="number" min={50} max={200} value={perms} onChange={e => setPerms(parseInt(e.target.value || '0', 10))} style={{ width: 80, marginLeft: 6 }} /></label>
            <label>{t('dumb.tau')} <input type="number" min={10} step={10} value={tau} onChange={e => setTau(parseInt(e.target.value || '0', 10))} style={{ width: 100, marginLeft: 6 }} /></label>
            <label>{t('dumb.fdr')}
              <select value={fdr} onChange={e => setFdr(e.target.value as any)} style={{ marginLeft: 6 }}>
                <option value="bh">BH</option>
                <option value="by">BY</option>
              </select>
            </label>
            <label><input type="checkbox" checked={useTe} onChange={e => setUseTe(e.target.checked)} /> {t('dumb.te')}</label>
            <label><input type="checkbox" checked={permJ} onChange={e => setPermJ(e.target.checked)} /> {t('dumb.jaccPerm')}</label>
          </div>
        </Card>
        {isLoading && <div className="muted">{t('dumb.loading')}</div>}
        {error && <div className="pill err">{t('dumb.error')}</div>}
        {data?.meta?.nbs_p != null && (
          <Alert
            type={data.meta.nbs_p <= 0.05 ? 'success' : 'warning'}
            showIcon
            message={
              <span>
                {t('dumb.nbs', { value: data.meta.nbs_p.toFixed(3) })}
                {data.meta.efp_upper != null && (
                  <>
                    {' · '}
                    {t('dumb.efp', { value: Number(data.meta.efp_upper).toFixed(1) })}
                  </>
                )}
              </span>
            }
            style={{ marginBottom: 12 }}
          />
        )}
        {cards.length > 0 && (
          <div className="row">
            {cards.map((p, idx) => {
              const cls = classify(p)
              return (
                <div className="card" key={`${p.u}-${p.v}-${idx}`} style={{ flex: 1, ...cardStyle(cls) }}>
                  <div className="title">{t('dumb.card.rank', { rank: idx + 1, a: p.u, b: p.v })}</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                    <span className="pill ok">{t('dumb.badge.matches', { count: p.agree_count })}</span>
                    <span className="pill">{t('dumb.badge.score', { score: p.score.toFixed(3) })}</span>
                    {typeof p.lag_seconds === 'number' && <span className="pill">{t('dumb.badge.lag', { lag: Math.round(p.lag_seconds) })}</span>}
                    {p.leader && <span className="pill warn">{t('dumb.badge.leader', { leader: p.leader })}</span>}
                    {verdictLabel(cls)}
                  </div>
                  <div className="muted" style={{ marginBottom: 8 }}>{t('dumb.explanation', { text: p.reason })}</div>
                  <div className="muted">{t('dumb.metrics')}</div>
                  <ul style={{ marginTop: 6, lineHeight: 1.6 }}>
                    <li>{t('pairs.metric.q')}: {p.metrics.q == null ? '-' : p.metrics.q.toFixed(3)}</li>
                    <li>{t('pairs.metric.absXcorr')}: {p.metrics.xcorr == null ? '-' : Math.abs(p.metrics.xcorr).toFixed(3)}</li>
                    <li>{t('pairs.metric.absSpearman')}: {p.metrics.spearman == null ? '-' : Math.abs(p.metrics.spearman).toFixed(3)}</li>
                    <li>{t('pairs.metric.jaccard')}: {p.metrics.jaccard == null ? '-' : p.metrics.jaccard.toFixed(3)}</li>
                    <li>{t('pairs.metric.jaccardQ')}: {p.metrics.jaccard_q == null ? '-' : p.metrics.jaccard_q.toFixed(3)}</li>
                    <li>{t('pairs.metric.mcc')}: {p.metrics.mcc == null ? '-' : p.metrics.mcc.toFixed(3)}</li>
                    <li>{t('pairs.metric.kappa')}: {p.metrics.kappa == null ? '-' : p.metrics.kappa.toFixed(3)}</li>
                    <li>{t('pairs.metric.ochiai')}: {p.metrics.ochiai == null ? '-' : p.metrics.ochiai.toFixed(3)}</li>
                    <li>{t('pairs.metric.overlap')}: {p.metrics.overlap == null ? '-' : p.metrics.overlap.toFixed(3)}</li>
                    <li>{t('pairs.metric.esi')}: {p.metrics.esi == null ? '-' : p.metrics.esi.toFixed(3)}</li>
                    <li>{t('pairs.metric.te')}: {p.metrics.te == null ? '-' : p.metrics.te.toFixed(3)}; {t('pairs.metric.teq')}: {p.metrics.te_q == null ? '-' : p.metrics.te_q.toFixed(3)}</li>
                  </ul>
                </div>
              )
            })}
          </div>
        )}
        {cards.length === 0 && !isLoading && !error && (
          <div className="pill">{t('dumb.empty')}</div>
        )}
      </div>
    </>
  )
}
