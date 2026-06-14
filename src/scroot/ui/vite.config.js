import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Dev server: proxy API calls to FastAPI backend
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7432',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  }
});
