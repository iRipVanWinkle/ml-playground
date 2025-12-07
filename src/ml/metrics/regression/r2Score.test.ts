import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { r2Score } from './r2Score';

describe('r2Score', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4], [5]]);
            const yPred = tf.tensor2d([[1], [2], [3], [4], [5]]);

            const result = r2Score(yTrue, yPred);
            const r2Value = result.dataSync()[0];

            expect(r2Value).toBeCloseTo(1, 5);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should return 0 when model predicts mean for all samples', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4], [5]]);
            // Mean of yTrue is 3
            const yPred = tf.tensor2d([[3], [3], [3], [3], [3]]);

            const result = r2Score(yTrue, yPred);
            const r2Value = result.dataSync()[0];

            // When predictions are the mean, R² should be ~0
            expect(r2Value).toBeCloseTo(0, 2);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should return negative value for predictions worse than mean', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4], [5]]);
            // Predictions that are worse than just predicting the mean
            const yPred = tf.tensor2d([[5], [4], [3], [2], [1]]);

            const result = r2Score(yTrue, yPred);
            const r2Value = result.dataSync()[0];

            // Reversed predictions should give negative R²
            expect(r2Value).toBeLessThan(0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct R² for partial fit', () => {
            const yTrue = tf.tensor2d([[1], [2], [3], [4], [5]]);
            const yPred = tf.tensor2d([[1.5], [2.5], [3], [3.5], [4.5]]);

            const result = r2Score(yTrue, yPred);
            const r2Value = result.dataSync()[0];

            // SS_res = (1-1.5)² + (2-2.5)² + (3-3)² + (4-3.5)² + (5-4.5)²
            //        = 0.25 + 0.25 + 0 + 0.25 + 0.25 = 1.0
            // Mean of yTrue = 3
            // SS_tot = (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²
            //        = 4 + 1 + 0 + 1 + 4 = 10
            // R² = 1 - (1.0 / 10) = 0.9
            expect(r2Value).toBeCloseTo(0.9, 2);

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

            const result = r2Score(yTrue, yPred);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });
});
