import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", sourcemap: true },
  server: {
    // In production the API and the app share an origin, because uvicorn serves
    // both. Proxying in development keeps that true, so there is no CORS to
    // configure and no second base URL to get wrong.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
