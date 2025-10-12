export type TensorBackend = 'auto' | 'webgpu' | 'webgl' | 'cpu' | 'wasm';

export type SystemSettings = {
    backend: TensorBackend;
    randomSeed?: number;
};
