import { defineConfig } from 'astro/config';

// output:'static' → SSG build into dist/, hosted by aiohttp in production.
// dev server proxies the live backend (WS + config + drilldown) to :8000.
export default defineConfig({
  output: 'static',
  server: { port: 4321 },
  vite: {
    server: {
      proxy: {
        '/ws': { target: 'ws://localhost:8000', ws: true },
        '/config': 'http://localhost:8000',
        '/api': 'http://localhost:8000',
      },
    },
  },
});
