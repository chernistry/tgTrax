import Link from 'next/link'
import { useRouter } from 'next/router'
import { Layout, Menu, Segmented, Typography } from 'antd'
import { LineChartOutlined, PartitionOutlined, TableOutlined, DashboardOutlined, StarOutlined } from '@ant-design/icons'
import { useI18n } from '../lib/i18n'

export default function Navbar() {
  const router = useRouter()
  const pathname = router.pathname
  const baseQuery = router.query || {}
  const { t, lang, setLang } = useI18n()
  const makeHref = (since: string) => ({ pathname, query: { ...baseQuery, since } })
  const items = [
    { key: '/', icon: <DashboardOutlined />, label: <Link href="/">{t('nav.dashboard')}</Link> },
    { key: '/pairs', icon: <TableOutlined />, label: <Link href="/pairs">{t('nav.pairs')}</Link> },
    { key: '/graph', icon: <PartitionOutlined />, label: <Link href="/graph">{t('nav.graph')}</Link> },
    { key: '/matrices', icon: <LineChartOutlined />, label: <Link href="/matrices">{t('nav.matrices')}</Link> },
    { key: '/for-idiots', icon: <StarOutlined />, label: <Link href="/for-idiots">{t('nav.dumb')}</Link> },
  ]
  const sinceValue = (baseQuery?.since as string) || 'start'
  return (
    <Layout.Header style={{ position: 'sticky', top: 0, zIndex: 100, width:'100%' }}>
      <div className="container" style={{ display:'flex', alignItems:'center', gap:16 }}>
        <Typography.Text style={{ color:'#cfe1f5', fontWeight:700 }}>tgTrax</Typography.Text>
        <Menu
          mode="horizontal"
          theme="dark"
          selectedKeys={[pathname]}
          items={items}
          style={{ flex:1, minWidth: 0 }}
        />
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <Segmented
            size="small"
            value={lang}
            onChange={val => setLang(val as 'en' | 'ru')}
            options={[
              { label: t('nav.lang.en'), value: 'en' },
              { label: t('nav.lang.ru'), value: 'ru' },
            ]}
          />
          <Typography.Text type="secondary">{t('nav.range')}</Typography.Text>
          <Segmented
            value={sinceValue}
            onChange={(val)=> router.push(makeHref(String(val)))}
            options={[
              { label:t('nav.range.start'), value:'start' },
              { label:t('nav.range.24h'), value:'24h' },
              { label:t('nav.range.7d'), value:'7d' },
              { label:t('nav.range.30d'), value:'30d' },
              { label:t('nav.range.all'), value:'all' },
            ]}
          />
        </div>
      </div>
    </Layout.Header>
  )
}
