import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config for the robustness dashboard.
// In development, /api requests are proxied to the FastAPI backend on :8000
// so the React dev server and the API can be run independently.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
