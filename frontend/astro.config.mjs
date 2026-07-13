import { defineConfig } from "astro/config";
import solid from "@astrojs/solid-js";

export default defineConfig({
  integrations: [solid()],
  output: "static",
  vite: {
    server: {
      proxy: {
        "/api": "http://127.0.0.1:5000",
      },
    },
  },
});
