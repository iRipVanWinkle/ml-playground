import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
    plugins: [react(), tailwindcss()],
    worker: { format: 'es' },
    base: process.env.NODE_ENV === 'production' ? '/ml-playground/' : '/',
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
});
