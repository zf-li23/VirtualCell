import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * GitHub Pages: set BASE env var to '/VirtualCell/' (or your repo name)
 * Local dev:    leave unset → defaults to '/'
 */
const base = process.env.BASE || '/'

// https://vite.dev/config/
export default defineConfig({
  base,
  plugins: [react()],
  build: {
    target: 'es2023',
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/react-dom') || id.includes('node_modules/react/')) {
            return 'react-vendor'
          }
          if (id.includes('node_modules/react-markdown') || id.includes('node_modules/remark-gfm')) {
            return 'markdown-vendor'
          }
          if (id.includes('node_modules/react-syntax-highlighter')) {
            return 'syntax-highlighter'
          }
        },
      },
    },
  },
})