import useSWR from 'swr'
import Navbar from '../components/Navbar'
import { apiBase, fetchJSON } from '../lib/api'
import { useMemo, useState } from 'react'
import { useRouter } from 'next/router'
import dynamic from 'next/dynamic'
import { useI18n } from '../lib/i18n'

const GraphCanvas = dynamic(() => import('reagraph').then(m => m.GraphCanvas), { ssr: false }) as any

type Node = { id: string, community?: number }
type Edge = { source: string, target: string, weight?: number, corr?: number|null, crosscorr?: number|null, qval?: number|null, jaccard?: number|null }
type Resp = { nodes: Node[], edges: Edge[] }

export default function GraphPage() {
  const router = useRouter()
  const since = (router.query?.since as string) || 'start'
  const [method, setMethod] = useState<'xcorr'|'spearman'|'pearson'>('xcorr')
  const [resid, setResid] = useState(false)
  const { t } = useI18n()
  const url = `${apiBase()}/api/graph/combined?layout=spring&corr_threshold=0.3&jacc_threshold=0.18&q_threshold=0.05&since=${encodeURIComponent(since)}&method=${method}&residualize=${resid}`
  const { data, error, isLoading } = useSWR<Resp>(url, fetchJSON)

  const { nodes, edges } = useMemo(() => {
    if (!data) return { nodes: [], edges: [] }
    const palette = ['#4cc9f0','#f72585','#7209b7','#3a0ca3','#4361ee','#2dd4bf','#f59e0b','#10b981']
    const nodes = data.nodes.map(n => ({ id: n.id, label: n.id, fill: palette[(n.community ?? 0) % palette.length], data: { community: n.community } }))
    const edges = data.edges.map((e, i) => ({
      id: `${e.source}__${e.target}__${i}`,
      source: e.source,
      target: e.target,
      label: (()=>{
        const parts:string[] = []
        if (typeof e.corr === 'number') parts.push(`r=${e.corr.toFixed(2)}`)
        if (typeof e.crosscorr === 'number') parts.push(`xc=${e.crosscorr.toFixed(2)}`)
        if (typeof e.qval === 'number') parts.push(`q=${e.qval.toFixed(2)}`)
        if (typeof e.jaccard === 'number') parts.push(`J=${e.jaccard.toFixed(2)}`)
        return parts.join(' · ')
      })(),
      size: Math.max(1.5, Math.min(6, (e.weight ?? 0.2) * 10)),
      data: e
    }))
    return { nodes, edges }
  }, [data])

  return (
    <>
      <Navbar />
      <div className="container">
        <h1 className="title">{t('graph.title')}</h1>
        <p className="muted">{t('graph.subtitle')}</p>
        <div className="card" style={{marginBottom:16}}>
          <div className="muted">{t('graph.display')}</div>
          <div style={{display:'flex', gap:8, marginTop:8, alignItems:'center', flexWrap:'wrap'}}>
            <label>{t('graph.metric')}
              <select value={method} onChange={e=>setMethod(e.target.value as any)} style={{marginLeft:6}}>
                <option value="xcorr">{t('graph.metric.xcorr')}</option>
                <option value="spearman">{t('graph.metric.spearman')}</option>
                <option value="pearson">{t('graph.metric.pearson')}</option>
              </select>
            </label>
            <label><input type="checkbox" checked={resid} onChange={e=>setResid(e.target.checked)} /> {t('graph.residualize')}</label>
          </div>
        </div>
        {isLoading && <div className="muted">{t('common.loading')}</div>}
        {error && <div className="pill err">{t('common.loadError')}</div>}
        {data && (
          <div style={{height:'70vh', border:'1px solid #333', position:'relative', overflow:'hidden', backgroundColor:'#1a1a1a'}}>
            <GraphCanvas
              nodes={nodes}
              edges={edges}
              layoutType="forceDirected2d"
              labelType="auto"
              edgeArrowPosition="none"
              draggable
              zoomable
              width={undefined}
              height={undefined}
            />
          </div>
        )}
      </div>
    </>
  )
}
