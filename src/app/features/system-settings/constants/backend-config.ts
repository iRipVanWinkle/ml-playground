export const BACKEND_LABELS: Record<string, string> = {
    webgpu: 'WebGPU',
    webgl: 'WebGL',
    cpu: 'CPU',
    wasm: 'WASM',
};

export const AVAILABLE_BACKENDS = [
    { value: 'webgpu', label: BACKEND_LABELS['webgpu'] },
    { value: 'webgl', label: BACKEND_LABELS['webgl'] },
    { value: 'cpu', label: BACKEND_LABELS['cpu'] },
    { value: 'wasm', label: BACKEND_LABELS['wasm'] },
];
