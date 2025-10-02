import useSWR from 'swr'
import Navbar from '../components/Navbar'
import { apiBase, fetchJSON, postJSON } from '../lib/api'
import { useState, useCallback } from 'react'
import { useI18n } from '../lib/i18n'

type SummaryResp = { summary: Record<string, any>, users: string[] }
type ConfigResp = { target_users_env: string[], db_path: string, session_path?: string }
type StatusResp = { running: boolean, pid?: number|null }
type StartResp = { started: boolean, pid?: number, message?: string, error?: string }
type LogsResp = { log: string }
type AuthStatus = { ok: boolean, authorized?: boolean, error?: string }

export default function Home() {
  const params = new URLSearchParams(typeof window !== 'undefined' ? window.location.search : '')
  const since = params.get('since') || 'start'
  const { data, error, isLoading } = useSWR<SummaryResp>(`${apiBase()}/api/summary?since=${encodeURIComponent(since)}`, fetchJSON)
  const { data: cfg } = useSWR<ConfigResp>(`${apiBase()}/api/config`, fetchJSON)
  const { data: status, mutate: mutateStatus } = useSWR<StatusResp & { started_at?: string }>(`${apiBase()}/api/stack/status`, fetchJSON, { refreshInterval: 2000 })
  const { data: logs, mutate: mutateLogs } = useSWR<LogsResp>(`${apiBase()}/api/stack/logs?lines=120`, fetchJSON, { refreshInterval: status?.running ? 3000 : 0 })
  const { data: auth } = useSWR<AuthStatus>(`${apiBase()}/api/auth/status`, fetchJSON)
  const { t } = useI18n()

  const [phone, setPhone] = useState('')
  const [code, setCode] = useState('')
  const [password, setPassword] = useState('')
  const [authMsgKey, setAuthMsgKey] = useState<string | undefined>()
  const [authMsgCustom, setAuthMsgCustom] = useState<string | undefined>()
  const [starting, setStarting] = useState(false)
  const [stopping, setStopping] = useState(false)

  const authMsg = authMsgCustom ?? (authMsgKey ? t(authMsgKey) : undefined)

  const startTracker = useCallback(async () => {
    setStarting(true)
    try {
      await postJSON<StartResp>(`${apiBase()}/api/stack/start`)
      await mutateStatus()
    } catch (e) {
      console.error(e)
    } finally {
      setStarting(false)
    }
  }, [mutateStatus])

  const stopTracker = useCallback(async () => {
    setStopping(true)
    try {
      await postJSON(`${apiBase()}/api/stack/stop`)
      await mutateStatus()
    } catch (e) {
      console.error(e)
    } finally {
      setStopping(false)
    }
  }, [mutateStatus])

  const handleSendCode = useCallback(async () => {
    try {
      setAuthMsgKey(undefined)
      setAuthMsgCustom(undefined)
      await postJSON(`${apiBase()}/api/auth/send_code`, { phone: phone || undefined })
      setAuthMsgKey('home.stack.codeSent')
    } catch (e: any) {
      console.error(e)
      setAuthMsgCustom(e?.message ?? t('common.loadError'))
    }
  }, [phone, t])

  const handleVerify = useCallback(async () => {
    try {
      setAuthMsgKey(undefined)
      setAuthMsgCustom(undefined)
      const r: any = await postJSON(`${apiBase()}/api/auth/verify`, { phone: phone || undefined, code, password: password || undefined })
      if (r.need_2fa) setAuthMsgKey('home.stack.verifyNeed2fa')
      else if (r.ok) setAuthMsgKey('home.stack.verifyOk')
      else setAuthMsgCustom(t('common.loadError'))
    } catch (e: any) {
      console.error(e)
      setAuthMsgCustom(e?.message ?? t('common.loadError'))
    }
  }, [phone, code, password, t])

  return (
    <>
      <Navbar />
      <div className="container">
        <h1 className="title">{t('home.title')}</h1>
        <p className="muted">{t('home.subtitle')}</p>
        {isLoading && <div className="muted">{t('common.loading')}</div>}
        {error && <div className="pill err">{t('common.loadError')}</div>}
        {data && (
          <div className="row">
            <div className="card">
              <div className="muted">{t('home.card.users')}</div>
              <div style={{ fontSize: '2rem', fontWeight: 700 }}>{data.users?.length ?? 0}</div>
            </div>
            <div className="card">
              <div className="muted">{t('home.card.periods')}</div>
              <div style={{ fontSize: '2rem', fontWeight: 700 }}>{data.summary?.num_resampled_periods ?? 0}</div>
            </div>
            <div className="card">
              <div className="muted">{t('home.card.duration')}</div>
              <div style={{ fontSize: '1.2rem' }}>{data.summary?.total_duration_analyzed ?? '-'}</div>
            </div>
          </div>
        )}
        <div className="row" style={{ marginTop: 16 }}>
          <div className="card">
            <div className="title">{t('home.stack.title')}</div>
            <div className="muted">{t('home.stack.targetUsers')}</div>
            <div style={{ marginBottom: 8 }}>{cfg?.target_users_env?.length ? cfg.target_users_env.join(', ') : t('common.none')}</div>
            <div className="muted">{t('home.stack.status')}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8 }}>
              <span className={`pill ${status?.running ? 'ok' : 'warn'}`}>
                {status?.running ? t('home.stack.running', { pid: status?.pid ?? '' }) : t('home.stack.stopped')}
              </span>
              {!status?.running ? (
                <button
                  onClick={startTracker}
                  disabled={starting}
                  style={{ padding: '6px 10px', borderRadius: 6, background: '#4cc9f0', border: 'none', color: '#001', fontWeight: 700 }}
                >
                  {starting ? t('home.stack.starting') : t('home.stack.start')}
                </button>
              ) : (
                <button
                  onClick={stopTracker}
                  disabled={stopping}
                  style={{ padding: '6px 10px', borderRadius: 6, background: '#ef476f', border: 'none', color: 'white', fontWeight: 700 }}
                >
                  {stopping ? t('home.stack.stopping') : t('home.stack.stop')}
                </button>
              )}
            </div>
            {status?.started_at && (
              <div className="muted" style={{ marginTop: 12 }}>
                {t('home.stack.startedAt')} <span style={{ color: '#cfe1f5' }}>{status.started_at}</span>
              </div>
            )}
            {cfg?.db_path && (
              <div className="muted" style={{ marginTop: 6 }}>
                {t('home.stack.db')} <span style={{ color: '#cfe1f5' }}>{cfg.db_path}</span>
              </div>
            )}
            {cfg?.session_path && (
              <div className="muted" style={{ marginTop: 6 }}>
                {t('home.stack.session')} <span style={{ color: '#cfe1f5' }}>{cfg.session_path}</span>
              </div>
            )}
            <div className="muted" style={{ marginTop: 12 }}>{t('home.stack.auth')}</div>
            <div style={{ marginTop: 6 }}>
              <span className={`pill ${auth?.authorized ? 'ok' : 'warn'}`}>
                {auth?.authorized ? t('home.stack.auth.ok') : t('home.stack.auth.missing')}
              </span>
            </div>
            {!auth?.authorized && (
              <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
                <div>
                  <label>
                    {t('home.stack.phone')}
                    <input placeholder="+123..." value={phone} onChange={e => setPhone(e.target.value)} style={{ marginLeft: 6 }} />
                  </label>
                  <button
                    onClick={handleSendCode}
                    style={{ marginLeft: 8, padding: '4px 8px', borderRadius: 6, background: '#1f2c47', border: '1px solid #2b3e61', color: '#cfe1f5' }}
                  >
                    {t('home.stack.sendCode')}
                  </button>
                </div>
                <div>
                  <label>
                    {t('home.stack.code')}
                    <input placeholder="xxxxxx" value={code} onChange={e => setCode(e.target.value)} style={{ marginLeft: 6 }} />
                  </label>
                  <label style={{ marginLeft: 12 }}>
                    {t('home.stack.password')}
                    <input type="password" placeholder="••••" value={password} onChange={e => setPassword(e.target.value)} style={{ marginLeft: 6 }} />
                  </label>
                  <button
                    onClick={handleVerify}
                    style={{ marginLeft: 8, padding: '4px 8px', borderRadius: 6, background: '#1f2c47', border: '1px solid #2b3e61', color: '#cfe1f5' }}
                  >
                    {t('home.stack.verify')}
                  </button>
                </div>
                {authMsg && <div className="muted">{authMsg}</div>}
              </div>
            )}
          </div>
          <div className="card" style={{ flex: 2 }}>
            <div className="title">{t('home.stack.logs')}</div>
            <div>
              <button
                onClick={() => mutateLogs()}
                style={{ padding: '4px 8px', borderRadius: 6, background: '#1f2c47', border: '1px solid #2b3e61', color: '#cfe1f5' }}
              >
                {t('common.refresh')}
              </button>
            </div>
            <pre style={{ whiteSpace: 'pre-wrap', background: '#0f182a', padding: 10, borderRadius: 6, maxHeight: 260, overflow: 'auto', marginTop: 8 }}>
{logs?.log ?? t('home.stack.logsEmpty')}
            </pre>
          </div>
        </div>
      </div>
    </>
  )
}
