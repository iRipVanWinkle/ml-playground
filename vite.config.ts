import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
    plugins: [react(), tailwindcss()],
    worker: { format: 'es' },
    base: process.env.NODE_ENV === 'production' ? '/ml-playground/' : '/',
    define: {
        global: 'globalThis', // fix for plotly.js (via has-hover)
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    'ui-radix': [
                        '@radix-ui/react-tabs',
                        '@radix-ui/react-select',
                        '@radix-ui/react-slider',
                        '@radix-ui/react-switch',
                        '@radix-ui/react-checkbox',
                        '@radix-ui/react-radio-group',
                        '@radix-ui/react-dropdown-menu',
                        '@radix-ui/react-tooltip',
                        '@radix-ui/react-hover-card',
                        '@radix-ui/react-separator',
                        '@radix-ui/react-progress',
                        '@radix-ui/react-label',
                        '@radix-ui/react-slot',
                    ],
                },
            },
        },
    },
});
