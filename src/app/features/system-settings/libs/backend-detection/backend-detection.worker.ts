/* eslint-disable @typescript-eslint/no-unused-vars */
import { getBackend, ready } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';

type BackendDetectionMessage = {
    type: 'detect-backends';
    requestId?: string;
};

type BackendInfo = {
    supported: string[];
    current?: string;
};

type ResponseMessage = {
    type: string;
    payload: BackendInfo;
    requestId?: string;
};

self.addEventListener('message', async (event: MessageEvent<BackendDetectionMessage>) => {
    const { type, requestId } = event.data;

    if (type === 'detect-backends') {
        try {
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

            const response: ResponseMessage = {
                type: 'success',
                payload: { supported, current },
                requestId,
            };

            self.postMessage(response);
        } catch (error) {
            const response: ResponseMessage = {
                type: 'error',
                payload: { supported: [], current: undefined },
                requestId,
            };

            self.postMessage(response);
        }
    }
});
