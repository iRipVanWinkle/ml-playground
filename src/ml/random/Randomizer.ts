import {
    gather,
    linspace,
    randomNormal,
    randomUniform,
    tidy,
    topk,
    type Rank,
    type ShapeMap,
    type Tensor,
} from '@tensorflow/tfjs';
import { assert } from '../utils';

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
        return randomUniform(shape, minval, maxval, dtype, Randomizer.mergeSeed(seed));
    }

    static randomUniqueNumber<R extends Rank>(
        shape: ShapeMap[R],
        minval?: number,
        maxval?: number,
        dtype: DataType = 'float32',
        seed?: number,
        poolSize = 10000,
    ): Tensor<R> {
        const numValues = shape.reduce((a, b) => a * b);

        assert(
            numValues < poolSize,
            `Number of unique values (${numValues}) exceeds pool size (${poolSize}). Increase poolSize.`,
        );

        return tidy(() => {
            const pool = linspace(minval ?? 0, maxval ?? 0, poolSize); // shape [poolSize]

            const randomKeys = randomUniform([poolSize], 0, 1, 'float32', seed);

            const sortedIndices = topk(randomKeys, poolSize, true).indices;

            const uniqueValues = gather(pool, sortedIndices.slice([0], [numValues]));

            const uniqueTensor = uniqueValues.reshape(shape);

            if (dtype === 'int32') {
                return uniqueTensor.floor().toInt();
            }

            return uniqueTensor;
        }) as Tensor<R>;
    }

    static randomNormal<R extends Rank>(
        shape: ShapeMap[R],
        mean?: number,
        stddev?: number,
        dtype: DataType = 'float32',
        seed?: number,
    ): Tensor<R> {
        return randomNormal(shape, mean, stddev, dtype, Randomizer.mergeSeed(seed));
    }

    static shuffle<T>(array: T[], seed?: number) {
        const mergedSeed = Randomizer.mergeSeed(seed);
        const rand = mulberry32(mergedSeed);

        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }

        return array;
    }

    private static mergeSeed(seed?: number): number | undefined {
        if (seed !== undefined && Randomizer.seed !== undefined) {
            return seed + Randomizer.seed;
        }
        return seed ?? Randomizer.seed;
    }
}

// Simple seeded PRNG (Mulberry32)
function mulberry32(seed = Date.now()) {
    return function () {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
