import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { generateFullPolynomialFeatures } from './generateFullPolynomialFeatures';

const sortArray = (array: number[][]): number[][] => {
    return array.map((a) => a.sort((a, b) => a - b));
};

describe('generatePolynomialFeatures', () => {
    const normalizeFunction = (x: tf.Tensor2D) => x; // No normalization for simplicity

    it('should generate polynomial features of degree 2', () => {
        const data = tf.tensor2d([[1, 2, 3, 4]]);
        const degree = 2;

        const result = generateFullPolynomialFeatures(data, degree, normalizeFunction)!;
        const expectedShape = [1, 10]; // 1 samples, 10 features: x1^2, x2^2, x1*x2

        expect(result.shape).toEqual(expectedShape);
        expect(sortArray(result.arraySync())).toEqual([[1, 2, 3, 4, 4, 6, 8, 9, 12, 16]]);
    });

    it('should generate polynomial features of degree 3', () => {
        const data = tf.tensor2d([
            [1, 2],
            [3, 4],
        ]);
        const degree = 3;

        const result = generateFullPolynomialFeatures(data, degree, normalizeFunction)!;
        const expectedShape = [2, 7]; // 2 samples, 7 features: x1^2, x2^2, x1*x2, x1^3, x2^3, x1^2*x2, x1*x2^2

        expect(result.shape).toEqual(expectedShape);
        expect(sortArray(result.arraySync())).toEqual([
            [1, 1, 2, 2, 4, 4, 8],
            [9, 12, 16, 27, 36, 48, 64],
        ]);
    });

    it('should keep memory clear', () => {
        const data = tf.tensor2d([[1, 2, 3, 4]]);
        const degree = 2;

        const prevNumTensors = tf.memory().numTensors;

        generateFullPolynomialFeatures(data, degree, normalizeFunction);
        const expectedNumTensors = prevNumTensors + 1;

        expect(tf.memory().numTensors).toEqual(expectedNumTensors);
    });
});
