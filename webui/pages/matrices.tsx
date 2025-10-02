import useSWR from 'swr'
import Navbar from '../components/Navbar'
import { apiBase, fetchJSON } from '../lib/api'
import dynamic from 'next/dynamic'
import { useMemo, useState } from 'react'
import { useRouter } from 'next/router'
import { useI18n } from '../lib/i18n'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any

type MatrixResp = { matrix: Record<string, Record<string, number|null>>, meta: Record<string, any> }

function toHeatmapData(m: Record<string, Record<string, number|null>>) {
  const rows = Object.keys(m)
  const cols = rows.length ? Object.keys(m[rows[0]]) : []
  const z = rows.map(r => cols.map(c => m[r]?.[c] ?? null))
  return { rows, cols, z }
}

export default function Matrices() {
  const router = useRouter()
  const since = (router.query?.since as string) || 'start'
  const [metric, setMetric] = useState<'spearman'|'pearson'|'jaccard'|'mcc'|'kappa'|'ochiai'|'overlap'|'crosscorr_max'|'crosscorr_qvals'>('spearman')
  const [resid, setResid] = useState(false)
  const { t } = useI18n()
  const url = `${apiBase()}/api/matrices?since=${encodeURIComponent(since)}&metric=${metric}&residualize=${resid}`
  const { data, error, isLoading } = useSWR<MatrixResp>(url, fetchJSON)
  const hm = useMemo(() => data ? toHeatmapData(data.matrix) : null, [data])
  return (
    <>
      <Navbar />
      <div className="container">
        <h1 className="title">{t('matrices.title')}</h1>
        <p className="muted">{t('matrices.subtitle')}</p>
        <div className="card" style={{marginBottom:16}}>
          <div className="muted">{t('matrices.metric')}</div>
          <div style={{display:'flex', gap:12, alignItems:'center', marginTop:8, flexWrap:'wrap'}}>
            <select value={metric} onChange={e=>setMetric(e.target.value as any)} style={{ padding:6, borderRadius:6, background:'#0f182a', color:'#cfe1f5', border:'1px solid #2b3e61'}}>
              <option value="spearman">{t('matrices.metric.spearman')}</option>
              <option value="pearson">{t('matrices.metric.pearson')}</option>
              <option value="jaccard">Jaccard</option>
              <option value="mcc">MCC</option>
              <option value="kappa">{t('matrices.metric.kappa')}</option>
              <option value="ochiai">Ochiai</option>
              <option value="overlap">{t('matrices.metric.overlap')}</option>
              <option value="crosscorr_max">{t('matrices.metric.xcorr')}</option>
              <option value="crosscorr_qvals">{t('matrices.metric.xcorrQ')}</option>
            </select>
            <label><input type="checkbox" checked={resid} onChange={e=>setResid(e.target.checked)} /> {t('matrices.residualize')}</label>
          </div>
        </div>
        {isLoading && <div className="muted">{t('common.loading')}</div>}
        {error && <div className="pill err">{t('common.loadError')}</div>}
        {hm && (
          <div className="card">
            <Plot
              data={[{ z: hm.z, x: hm.cols, y: hm.rows, type: 'heatmap', colorscale: 'Viridis', reversescale: false }]}
              layout={{ paper_bgcolor: '#111a2b', plot_bgcolor: '#111a2b', font: { color: '#cfe1f5' }, margin: {l:80,r:10,t:10,b:80}, xaxis: {tickangle: 45} }}
              style={{ width: '100%', height: '700px' }}
              config={{ displayModeBar: false }}
            />
          </div>
        )}
      </div>
    </>
  )
}
