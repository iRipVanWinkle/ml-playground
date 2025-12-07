import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { meanSquaredError } from './meanSquaredError';

describe('meanSquaredError', () => {
    describe('basic functionality', () => {
        it('should return 0 for perfect predictions', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = meanSquaredError(yTrue, yPred);
            const mseValue = result.dataSync()[0];

            expect(mseValue).toBe(0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct MSE for simple errors', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4]]);
            const yPred = tf.tensor2d([[2], [3], [4], [5]]);

            const result = meanSquaredError(yTrue, yPred);
            const mseValue = result.dataSync()[0];

            // Errors: (1-2)²=1, (2-3)²=1, (3-4)²=1, (4-5)²=1
            // MSE = (1 + 1 + 1 + 1) / 4 = 1
            expect(mseValue).toBe(1);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct MSE for mixed errors', () => {
            const yTrue = tf.tensor2d([[1.0], [2.0], [3.0]]);
            const yPred = tf.tensor2d([[1.5], [2.5], [3.5]]);

            const result = meanSquaredError(yTrue, yPred);
            const mseValue = result.dataSync()[0];

            // Errors: (1.0-1.5)²=0.25, (2.0-2.5)²=0.25, (3.0-3.5)²=0.25
            // MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
            expect(mseValue).toBeCloseTo(0.25, 5);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle negative differences correctly', () => {
            const yTrue = tf.tensor2d([[5], [6], [7]]);
            const yPred = tf.tensor2d([[3], [4], [5]]);

            const result = meanSquaredError(yTrue, yPred);
            const mseValue = result.dataSync()[0];

            // Errors: (5-3)²=4, (6-4)²=4, (7-5)²=4
            // MSE = (4 + 4 + 4) / 3 = 4
            expect(mseValue).toBe(4);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('sensitivity to outliers', () => {
        it('should penalize outliers heavily due to squaring', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [100]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = meanSquaredError(yTrue, yPred);
            const mseValue = result.dataSync()[0];

            // Errors: 0, 0, 0, (100-4)²=9216
            // MSE = 9216 / 4 = 2304
            expect(mseValue).toBe(2304);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('memory management', () => {
        it('should not leak memory during computation', () => {
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1], [2], [3]]);
            const initialTensors = tf.memory().numTensors;

            const result = meanSquaredError(yTrue, yPred);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });
});
