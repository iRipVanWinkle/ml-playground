import {
    randomNormal,
    randomUniform,
    type Rank,
    type ShapeMap,
    type Tensor,
} from '@tensorflow/tfjs';

type DataType = 'float32' | 'int32';

export class Randomizer {
    private static seed?: number = 42;

    private constructor() {}

    static setSeed(seed?: number) {
        Randomizer.seed = seed;
    }

    static randomUniform<R extends Rank>(
        shape: ShapeMap[R],
        minval?: number,
        maxval?: number,
        dtype: DataType = 'float32',
        seed?: number,
    ): Tensor<R> {
        return randomUniform(shape, minval, maxval, dtype, seed ?? Randomizer.seed);
    }

    static randomNormal<R extends Rank>(
        shape: ShapeMap[R],
        mean?: number,
        stddev?: number,
        dtype: DataType = 'float32',
        seed?: number,
    ): Tensor<R> {
        return randomNormal(shape, mean, stddev, dtype, seed ?? Randomizer.seed);
    }
}
