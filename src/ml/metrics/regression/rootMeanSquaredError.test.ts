import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { rootMeanSquaredError } from './rootMeanSquaredError';

describe('rootMeanSquaredError', () => {
    describe('basic functionality', () => {
        it('should return 0 for perfect predictions', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = rootMeanSquaredError(yTrue, yPred);
            const rmseValue = result.dataSync()[0];

            expect(rmseValue).toBe(0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct RMSE for mixed errors', () => {
            const yTrue = tf.tensor2d([[1.0], [2.0], [3.0]]);
            const yPred = tf.tensor2d([[1.5], [2.5], [3.5]]);

            const result = rootMeanSquaredError(yTrue, yPred);
            const rmseValue = result.dataSync()[0];

            // MSE = 0.25, RMSE = √0.25 = 0.5
            expect(rmseValue).toBeCloseTo(0.5, 5);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('sensitivity to outliers', () => {
        it('should penalize outliers but less than MSE due to sqrt', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [100]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = rootMeanSquaredError(yTrue, yPred);
            const rmseValue = result.dataSync()[0];

            // MSE = 2304, RMSE = √2304 = 48
            expect(rmseValue).toBe(48);

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

            const result = rootMeanSquaredError(yTrue, yPred);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });
});
