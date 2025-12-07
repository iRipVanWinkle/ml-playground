import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { meanAbsoluteError } from './meanAbsoluteError';

describe('meanAbsoluteError', () => {
    describe('basic functionality', () => {
        it('should return 0 for perfect predictions', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = meanAbsoluteError(yTrue, yPred);
            const maeValue = result.dataSync()[0];

            expect(maeValue).toBe(0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct MAE for simple errors', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4]]);
            const yPred = tf.tensor2d([[2], [3], [4], [5]]);

            const result = meanAbsoluteError(yTrue, yPred);
            const maeValue = result.dataSync()[0];

            // All errors are 1, so MAE = 1
            expect(maeValue).toBe(1);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct MAE for mixed errors', () => {
            const yTrue = tf.tensor2d([[1.5], [2.5], [3.5]]);
            const yPred = tf.tensor2d([[1.0], [2.0], [3.0]]);

            const result = meanAbsoluteError(yTrue, yPred);
            const maeValue = result.dataSync()[0];

            // Errors: |1.5-1.0|=0.5, |2.5-2.0|=0.5, |3.5-3.0|=0.5
            // MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
            expect(maeValue).toBeCloseTo(0.5, 5);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle negative differences correctly', () => {
            const yTrue = tf.tensor2d([[5], [6], [7]]);
            const yPred = tf.tensor2d([[3], [4], [5]]);

            const result = meanAbsoluteError(yTrue, yPred);
            const maeValue = result.dataSync()[0];

            // Errors: |5-3|=2, |6-4|=2, |7-5|=2
            // MAE = (2 + 2 + 2) / 3 = 2
            expect(maeValue).toBe(2);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('robustness to outliers', () => {
        it('should handle outliers without excessive penalty', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [100]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = meanAbsoluteError(yTrue, yPred);
            const maeValue = result.dataSync()[0];

            // Errors: 0, 0, 0, 96
            // MAE = 96 / 4 = 24
            expect(maeValue).toBe(24);

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

            const result = meanAbsoluteError(yTrue, yPred);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });
});
