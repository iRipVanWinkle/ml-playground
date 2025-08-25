import {
    zeros,
    ones,
    type Tensor2D,
    fill,
    randomUniform,
    randomNormal,
    concat,
    tidy,
} from '@tensorflow/tfjs';

export type ThetaInitializer = (shape: [number, number], withBias?: boolean) => Tensor2D;

function addBiasRow(X: Tensor2D): Tensor2D {
    const bias = zeros([1, X.shape[1]]);
    return concat([bias, X], 0) as Tensor2D;
}

/**
 * Zeros initializer
 */
export function zerosInitializer(): ThetaInitializer {
    return (shape, withBias = true) => zeros([withBias ? shape[0] + 1 : shape[0], shape[1]]);
}

/**
 * Ones initializer
 */
export function onesInitializer(): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const tensor = ones(shape) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * Constant initializer
 */
export function constantInitializer(value: number): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const tensor = fill(shape, value) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * Uniform initializer
 */
export function uniformInitializer(min: number, max: number): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const tensor = randomUniform(shape, min, max, 'float32', 42) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * Normal initializer
 */
export function normalInitializer(mean: number, stddev: number): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const tensor = randomNormal(shape, mean, stddev, 'float32', 42) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * Xavier/Glorot Uniform initializer
 */
export function xavierUniformInitializer(): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const [fanIn, fanOut] = shape;
            const limit = Math.sqrt(6 / (fanIn + fanOut));
            const tensor = randomUniform(shape, -limit, limit, 'float32', 42) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * Xavier/Glorot Normal initializer
 */
export function xavierNormalInitializer(): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const [fanIn, fanOut] = shape;
            const stddev = Math.sqrt(2 / (fanIn + fanOut));
            const tensor = randomNormal(shape, 0, stddev, 'float32', 42) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * He Uniform initializer
 */
export function heUniformInitializer(): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const fanIn = shape[0];
            const limit = Math.sqrt(6 / fanIn);
            const tensor = randomUniform(shape, -limit, limit, 'float32', 42) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}

/**
 * He Normal initializer
 */
export function heNormalInitializer(): ThetaInitializer {
    return (shape, withBias = true) =>
        tidy(() => {
            const fanIn = shape[0];
            const stddev = Math.sqrt(2 / fanIn);
            const tensor = randomNormal(shape, 0, stddev, 'float32', 42) as Tensor2D;
            return withBias ? addBiasRow(tensor) : tensor;
        });
}
