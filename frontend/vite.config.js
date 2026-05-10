import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite 基础配置，第一阶段只保证 React 页面可以启动。
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173
  }
});
