import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = env.VITE_PROXY_TARGET || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      allowedHosts: [
        'm8kg4cgo8kswcokskskcgkco.178.63.85.114.sslip.io',
        'openshorts.app',
        'www.openshorts.app'
      ],
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
        },
        '/videos': {
          target: proxyTarget,
          changeOrigin: true,
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
        }
      }
    }
  }
})
