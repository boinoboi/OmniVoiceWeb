import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const repo = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = process.env.BASE_PATH ?? (repo ? `/${repo}/` : '/')

export default defineConfig({
  base,
  plugins: [react()],
  worker: { format: 'es' },
  build: { target: 'esnext' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
