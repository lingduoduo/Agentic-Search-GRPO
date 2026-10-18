import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    cssCodeSplit: true,
    sourcemap: false,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: "react", test: /node_modules\/(react|react-dom)\// },
            { name: "icons", test: /node_modules\/lucide-react\// },
          ],
        },
      },
    },
  },
  server: {
    // Proxy every backend prefix the app calls (api.ts). The admin panels use
    // unprefixed routers (/admin, /analytics, /chat); without these the panels
    // silently fail in dev because requests hit the Vite server, not FastAPI.
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/health": "http://127.0.0.1:7860",
      "/auth": "http://127.0.0.1:7860",
      "/me": "http://127.0.0.1:7860",
      "/admin": "http://127.0.0.1:7860",
      "/analytics": "http://127.0.0.1:7860",
      "/chat": {
        target: "http://127.0.0.1:7860",
        // A page navigation asks for HTML; that is the SPA route, not the API.
        // API calls from the app send Accept: application/json and still proxy.
        bypass: (req) => (req.headers.accept?.includes("text/html") ? "/index.html" : undefined),
      },
      "/search": {
        target: "http://127.0.0.1:7860",
        bypass: (req) => (req.headers.accept?.includes("text/html") ? "/index.html" : undefined),
      },
      // Proxy keys match by prefix, so "/tool" below would otherwise swallow the
      // /tools page too. Listed first because the first matching key wins.
      "/tools": {
        target: "http://127.0.0.1:7860",
        bypass: (req) => (req.headers.accept?.includes("text/html") ? "/index.html" : undefined),
      },
      "/tool": "http://127.0.0.1:7860",
    },
  },
});
