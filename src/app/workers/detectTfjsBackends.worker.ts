/* eslint-disable @typescript-eslint/no-unused-vars */
import { getBackend, ready } from "@tensorflow/tfjs";
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';

(async () => {
    const supported: string[] = [];

    // CPU is always available
    supported.push('cpu');

    // Check for WASM support
    if (typeof WebAssembly !== 'undefined') {
        supported.push('wasm');
    }

    // Check for WebGL support
    try {
        const canvas = new OffscreenCanvas(1, 1);

        if (
            self.WebGLRenderingContext &&
            (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
        ) {
            supported.push('webgl');
        }
    } catch (_) {
        /* not supported */
    }

    // Check for WebGPU support
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
        supported.push('webgpu');
    }

    await ready();
    const current = getBackend();

    self.postMessage({ supported, current });
})();
