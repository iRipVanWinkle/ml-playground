import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { generateCosinusoidalFeatures } from './generateCosinusoidalFeatures';

const sortArray = (array: number[][]): number[][] => {
    return array.map((a) => a.sort((a, b) => a - b));
};

describe('generateCosinusoidalFeatures', () => {
    it('should generate sinusoidal features of degree 2', () => {
        const data = tf.tensor2d([[1, 2, 3, 4]]);
        const degree = 2;

        const result = generateCosinusoidalFeatures(data, degree);
        const expectedShape = [1, 8]; // 1 sample, 8 features: sin(x1), sin(x2), sin(x3), sin(x4), sin(2*x1), sin(2*x2), sin(2*x3), sin(2*x4)

        expect(result.shape).toEqual(expectedShape);
        expect(sortArray(result.arraySync())).toEqual([
            [
                expect.closeTo(-0.99),
                expect.closeTo(-0.6536),
                expect.closeTo(-0.6536),
                expect.closeTo(-0.4161),
                expect.closeTo(-0.4161),
                expect.closeTo(-0.1455),
                expect.closeTo(0.5403),
                expect.closeTo(0.9601),
            ],
        ]);
    });

    it('should generate sinusoidal features of degree 3', () => {
        const data = tf.tensor2d([
            [1, 2],
            [3, 4],
        ]);
        const degree = 3;

        const result = generateCosinusoidalFeatures(data, degree);
        const expectedShape = [2, 6]; // 2 samples, 6 features: sin(x1), sin(x2), sin(2*x1), sin(2*x2), sin(3*x1), sin(3*x2)

        expect(result.shape).toEqual(expectedShape);
        expect(sortArray(result.arraySync())).toEqual([
            [
                expect.closeTo(-0.99),
                expect.closeTo(-0.6536),
                expect.closeTo(-0.4161),
                expect.closeTo(-0.4161),
                expect.closeTo(0.5403),
                expect.closeTo(0.9601),
            ],
            [
                expect.closeTo(-0.99),
                expect.closeTo(-0.9111),
                expect.closeTo(-0.6536),
                expect.closeTo(-0.1455),
                expect.closeTo(0.8438),
                expect.closeTo(0.9602),
            ],
        ]);
    });

    it('should keep memory clear', () => {
        const data = tf.tensor2d([[1, 2, 3, 4]]);
        const degree = 2;

        const prevNumTensors = tf.memory().numTensors;

        generateCosinusoidalFeatures(data, degree);
        const expectedNumTensors = prevNumTensors + 1;

        expect(tf.memory().numTensors).toEqual(expectedNumTensors);
    });
});
