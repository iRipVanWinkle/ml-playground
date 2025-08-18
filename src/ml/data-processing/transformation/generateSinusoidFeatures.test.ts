import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { generateSinusoidalFeatures } from './generateSinusoidFeatures';

const sortArray = (array: number[][]): number[][] => {
    return array.map((a) => a.sort((a, b) => a - b));
};

describe('generateSinusoidalFeatures', () => {
    it('should generate sinusoidal features of degree 2', () => {
        const data = tf.tensor2d([[1, 2, 3, 4]]);
        const degree = 2;

        const result = generateSinusoidalFeatures(data, degree);
        const expectedShape = [1, 8]; // 1 sample, 8 features: sin(x1), sin(x2), sin(x3), sin(x4), sin(2*x1), sin(2*x2), sin(2*x3), sin(2*x4)

        expect(result.shape).toEqual(expectedShape);
        expect(sortArray(result.arraySync())).toEqual([
            [
                expect.closeTo(-0.7568),
                expect.closeTo(-0.7568),
                expect.closeTo(-0.2794),
                expect.closeTo(0.1411),
                expect.closeTo(0.8415),
                expect.closeTo(0.9093),
                expect.closeTo(0.9093),
                expect.closeTo(0.9894),
            ],
        ]);
    });

    it('should generate sinusoidal features of degree 3', () => {
        const data = tf.tensor2d([
            [1, 2],
            [3, 4],
        ]);
        const degree = 3;

        const result = generateSinusoidalFeatures(data, degree);
        const expectedShape = [2, 6]; // 2 samples, 6 features: sin(x1), sin(x2), sin(2*x1), sin(2*x2), sin(3*x1), sin(3*x2)

        expect(result.shape).toEqual(expectedShape);
        expect(sortArray(result.arraySync())).toEqual([
            [
                expect.closeTo(-0.7568),
                expect.closeTo(-0.2794),
                expect.closeTo(0.1411),
                expect.closeTo(0.8415),
                expect.closeTo(0.9093),
                expect.closeTo(0.9093),
            ],
            [
                expect.closeTo(-0.7568),
                expect.closeTo(-0.5366),
                expect.closeTo(-0.2794),
                expect.closeTo(0.1411),
                expect.closeTo(0.4121),
                expect.closeTo(0.9894),
            ],
        ]);
    });

    it('should keep memory clear', () => {
        const data = tf.tensor2d([[1, 2, 3, 4]]);
        const degree = 2;

        const prevNumTensors = tf.memory().numTensors;

        generateSinusoidalFeatures(data, degree);
        const expectedNumTensors = prevNumTensors + 1;

        expect(tf.memory().numTensors).toEqual(expectedNumTensors);
    });
});
