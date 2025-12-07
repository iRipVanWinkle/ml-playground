import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { residuals } from './residuals';

describe('residuals', () => {
    describe('basic functionality', () => {
        it('should return zeros for perfect predictions', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4]]);

            const result = residuals(yTrue, yPred);
            const residualValues = result.dataSync();

            expect(Array.from(residualValues)).toEqual([0, 0, 0, 0]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct residuals for mixed errors', () => {
            const yTrue = tf.tensor2d([[1], [5.5], [3], [7]]);
            const yPred = tf.tensor2d([[2], [3], [3.5], [10]]);

            const result = residuals(yTrue, yPred);
            const residualValues = result.dataSync();

            // residuals = [1-2, 5.5-3, 3-3.5, 7-10] = [-1, 2.5, -0.5, -3]
            expect(Array.from(residualValues)).toEqual([-1, 2.5, -0.5, -3]);

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

            const result = residuals(yTrue, yPred);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });
});
