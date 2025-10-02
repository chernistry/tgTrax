import type { AppProps } from 'next/app'
import '../styles/globals.css'
import 'antd/dist/reset.css'
import { ConfigProvider, theme as antdTheme, App as AntApp } from 'antd'
import { I18nProvider } from '../lib/i18n'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <I18nProvider>
      <ConfigProvider
        theme={{
          algorithm: antdTheme.darkAlgorithm,
          token: {
            colorPrimary: '#4cc9f0',
            colorInfo: '#4cc9f0',
            fontSize: 14,
            borderRadius: 8,
          },
        }}
      >
        <AntApp>
          <Component {...pageProps} />
        </AntApp>
      </ConfigProvider>
    </I18nProvider>
  )
}
