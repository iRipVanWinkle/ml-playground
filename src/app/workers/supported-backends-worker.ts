/* eslint-disable @typescript-eslint/no-unused-vars */
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

self.postMessage(supported);
