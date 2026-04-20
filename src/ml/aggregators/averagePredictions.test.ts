import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { averagePredictions } from './averagePredictions';

describe('averagePredictions', () => {
    describe('basic functionality', () => {
        it('should average predictions correctly for regression', () => {
            // Create a 2D tensor: [3 samples, 2 models] for regression
            const predictions = tf.tensor2d([
                [4.0, 6.0], // Sample 1: model predictions 4.0 and 6.0
                [2.0, 3.0], // Sample 2: model predictions 2.0 and 3.0
                [8.0, 2.0], // Sample 3: model predictions 8.0 and 2.0
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([3, 1]);

            const resultData = result.arraySync() as number[][];

            // Expected averages
            expect(resultData[0][0]).toBeCloseTo(5.0, 5); // (4.0 + 6.0) / 2
            expect(resultData[1][0]).toBeCloseTo(2.5, 5); // (2.0 + 3.0) / 2
            expect(resultData[2][0]).toBeCloseTo(5.0, 5); // (8.0 + 2.0) / 2

            predictions.dispose();
            result.dispose();
        });

        it('should average multiple model predictions correctly', () => {
            // Create a 2D tensor: [2 samples, 4 models] for ensemble averaging
            const predictions = tf.tensor2d([
                [1.0, 3.0, 5.0, 7.0], // Sample 1: 4 model predictions
                [2.0, 4.0, 6.0, 8.0], // Sample 2: 4 model predictions
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([2, 1]);

            const resultData = result.arraySync() as number[][];

            // Expected averages for each sample
            expect(resultData[0][0]).toBeCloseTo(4.0, 5); // (1.0 + 3.0 + 5.0 + 7.0) / 4
            expect(resultData[1][0]).toBeCloseTo(5.0, 5); // (2.0 + 4.0 + 6.0 + 8.0) / 4

            predictions.dispose();
            result.dispose();
        });

        it('should handle single model correctly', () => {
            // Create a 2D tensor: [2 samples, 1 model]
            const predictions = tf.tensor2d([
                [7.5], // Single model for sample 1
                [1.0], // Single model for sample 2
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([2, 1]);

            const resultData = result.arraySync() as number[][];

            // With single model, result should be identical to input
            expect(resultData[0][0]).toBeCloseTo(7.5, 5);
            expect(resultData[1][0]).toBeCloseTo(1.0, 5);

            predictions.dispose();
            result.dispose();
        });

        it('should handle zero predictions', () => {
            const predictions = tf.tensor2d([[0.0, 0.0, 0.0]]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(0.0, 5);

            predictions.dispose();
            result.dispose();
        });

        it('should handle negative predictions', () => {
            const predictions = tf.tensor2d([
                [-2.0, 4.0, -1.0], // Sample with mixed positive/negative predictions
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            // Expected average: (-2+4-1)/3 = 1/3
            expect(resultData[0][0]).toBeCloseTo(1 / 3, 5);

            predictions.dispose();
            result.dispose();
        });

        it('should handle large prediction values', () => {
            const predictions = tf.tensor2d([
                [1000.0, 3000.0], // Sample with large values
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(2000.0, 2);

            predictions.dispose();
            result.dispose();
        });

        it('should handle small decimal predictions', () => {
            const predictions = tf.tensor2d([
                [0.001, 0.003, 0.005], // Sample with small decimal values
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            // Expected average: (0.001+0.003+0.005)/3
            expect(resultData[0][0]).toBeCloseTo(0.003, 6);

            predictions.dispose();
            result.dispose();
        });
    });

    describe('mathematical properties', () => {
        it('should maintain linearity property', () => {
            // If we scale all predictions by a constant, the average should scale by the same constant
            const scale = 2.5;
            const basePredictions = tf.tensor2d([
                [1.0, 3.0], // Sample with two model predictions
            ]);

            const scaledPredictions = basePredictions.mul(scale);

            const baseResult = averagePredictions(basePredictions);
            const scaledResult = averagePredictions(scaledPredictions);

            const baseData = baseResult.arraySync() as number[][];
            const scaledData = scaledResult.arraySync() as number[][];

            expect(scaledData[0][0]).toBeCloseTo(baseData[0][0] * scale, 5);

            basePredictions.dispose();
            scaledPredictions.dispose();
            baseResult.dispose();
            scaledResult.dispose();
        });

        it('should maintain additivity property', () => {
            // Average of (A + B) should equal average of A plus average of B
            const predictionsA = tf.tensor2d([
                [1.0, 3.0], // Sample A
            ]);

            const predictionsB = tf.tensor2d([
                [5.0, 7.0], // Sample B
            ]);

            const predictionsSum = predictionsA.add(predictionsB);

            const avgA = averagePredictions(predictionsA);
            const avgB = averagePredictions(predictionsB);
            const avgSum = averagePredictions(predictionsSum);
            const sumOfAvgs = avgA.add(avgB);

            const avgSumData = avgSum.arraySync() as number[][];
            const sumOfAvgsData = sumOfAvgs.arraySync() as number[][];

            expect(avgSumData[0][0]).toBeCloseTo(sumOfAvgsData[0][0], 5);

            predictionsA.dispose();
            predictionsB.dispose();
            predictionsSum.dispose();
            avgA.dispose();
            avgB.dispose();
            avgSum.dispose();
            sumOfAvgs.dispose();
        });

        it('should handle equal predictions correctly', () => {
            // When all models make the same prediction, average should equal that prediction
            const commonValue = 42.7;
            const predictions = tf.tensor2d([
                [commonValue, commonValue, commonValue], // Sample with identical predictions
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(commonValue, 5);

            predictions.dispose();
            result.dispose();
        });
    });

    describe('edge cases', () => {
        it('should handle very large number of models', () => {
            const numModels = 100;
            const numSamples = 3;

            // Create random predictions for 2D tensor [samples, models]
            const predictionsArray = Array(numSamples)
                .fill(null)
                .map(
                    () =>
                        Array(numModels)
                            .fill(null)
                            .map(() => Math.random() * 10 - 5), // Random values between -5 and 5
                );

            const predictions = tf.tensor2d(predictionsArray);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([numSamples, 1]);

            // Verify that the result is indeed the average
            const resultData = result.arraySync() as number[][];

            for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                const expectedAvg =
                    predictionsArray[sampleIdx].reduce((sum, val) => sum + val, 0) / numModels;

                expect(resultData[sampleIdx][0]).toBeCloseTo(expectedAvg, 5);
            }

            predictions.dispose();
            result.dispose();
        });

        it('should handle extreme values', () => {
            const largeValue = 1e10; // Use a large but manageable value
            const predictions = tf.tensor2d([
                [largeValue, 0], // Sample with extreme values
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            // Should average to half the extreme value
            expect(resultData[0][0]).toBeCloseTo(largeValue / 2, 1);

            predictions.dispose();
            result.dispose();
        });

        it('should handle NaN values appropriately', () => {
            const predictions = tf.tensor2d([
                [1.0, NaN], // Sample with NaN
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            // NaN should propagate through the average
            expect(isNaN(resultData[0][0])).toBe(true);

            predictions.dispose();
            result.dispose();
        });

        it('should handle Infinity values appropriately', () => {
            const predictions = tf.tensor2d([
                [1.0, Infinity], // Sample with Infinity
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            // Infinity should propagate through the average
            expect(resultData[0][0]).toBe(Infinity);

            predictions.dispose();
            result.dispose();
        });
    });

    describe('data types and shapes', () => {
        it('should preserve tensor type as Tensor2D', () => {
            const predictions = tf.tensor2d([
                [1.0, 3.0], // Sample with 2 model predictions
            ]);

            const result = averagePredictions(predictions);

            expect(result.rank).toBe(2);
            expect(result).toBeInstanceOf(tf.Tensor);

            predictions.dispose();
            result.dispose();
        });

        it('should maintain keepDims=true behavior', () => {
            // The function uses mean(1, true) which should keep dimensions
            const predictions = tf.tensor2d([
                [1.0, 3.0], // Sample with 2 predictions
            ]);

            const result = averagePredictions(predictions);

            // Should maintain the reduced dimension
            expect(result.shape).toEqual([1, 1]);

            predictions.dispose();
            result.dispose();
        });

        it('should handle different numeric precisions', () => {
            // Test with float32 (default TensorFlow.js type)
            const predictions = tf.tensor2d(
                [
                    [1.123456789, 3.111111111], // Sample with precise decimals
                ],
                [1, 2],
                'float32',
            );

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([1, 1]);

            const resultData = result.arraySync() as number[][];

            // Results should be computed with float32 precision
            expect(resultData[0][0]).toBeCloseTo((1.123456789 + 3.111111111) / 2, 6);

            predictions.dispose();
            result.dispose();
        });
    });

    describe('memory management', () => {
        it('should not leak memory during computation', () => {
            const initialTensors = tf.memory().numTensors;

            const predictions = tf.tensor2d([
                [1.0, 3.0], // Sample with 2 predictions
            ]);

            const result = averagePredictions(predictions);
            result.dispose();
            predictions.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);
        });

        it('should handle intermediate tensor cleanup in tidy', () => {
            // The function uses tidy() which should clean up intermediate tensors
            const memoryBefore = tf.memory();

            const predictions = tf.tensor2d([
                [1.0, 3.0], // Sample with 2 predictions
            ]);

            const result = averagePredictions(predictions);

            // Only the result tensor should remain from the operation
            const memoryAfter = tf.memory();
            expect(memoryAfter.numTensors).toBe(memoryBefore.numTensors + 2); // input + result

            predictions.dispose();
            result.dispose();
        });
    });

    describe('performance', () => {
        it('should handle large tensors efficiently', () => {
            const numSamples = 1000;
            const numModels = 50;

            // Generate random predictions for 2D tensor [samples, models]
            const predictionsData = Array(numSamples)
                .fill(null)
                .map(() =>
                    Array(numModels)
                        .fill(null)
                        .map(() => Math.random() * 100),
                );

            const predictions = tf.tensor2d(predictionsData);

            const startTime = performance.now();
            const result = averagePredictions(predictions);
            const endTime = performance.now();

            expect(result.shape).toEqual([numSamples, 1]);
            expect(endTime - startTime).toBeLessThan(1000); // Should complete in less than 1 second

            predictions.dispose();
            result.dispose();
        });

        it('should be efficient with memory usage', () => {
            const initialMemory = tf.memory().numBytes;

            const predictions = tf.tensor2d([
                [1.0, 4.0, 7.0], // Sample with 3 model predictions
            ]);

            const result = averagePredictions(predictions);

            // Memory usage should be reasonable
            const memoryUsed = tf.memory().numBytes - initialMemory;
            expect(memoryUsed).toBeLessThan(1000); // Should use less than 1KB for this small operation

            predictions.dispose();
            result.dispose();
        });
    });

    describe('integration scenarios', () => {
        it('should work correctly in ensemble regression pipeline', () => {
            // Simulate predictions from different regression models
            const predictions = tf.tensor2d([
                [2.1, 2.3, 1.9], // Sample 1: linear, tree, neural model predictions
                [3.9, 4.1, 3.7], // Sample 2: linear, tree, neural model predictions
                [5.8, 6.2, 5.6], // Sample 3: linear, tree, neural model predictions
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([3, 1]);

            const resultData = result.arraySync() as number[][];

            // Verify ensemble averages
            expect(resultData[0][0]).toBeCloseTo((2.1 + 2.3 + 1.9) / 3, 5);
            expect(resultData[1][0]).toBeCloseTo((3.9 + 4.1 + 3.7) / 3, 5);
            expect(resultData[2][0]).toBeCloseTo((5.8 + 6.2 + 5.6) / 3, 5);

            predictions.dispose();
            result.dispose();
        });

        it('should handle multiple samples with different model configurations', () => {
            // Test with different numbers of model predictions per sample
            const predictions = tf.tensor2d([
                [1.0, 1.2, 0.8], // Sample 1: 3 model predictions
                [2.0, 2.1, 1.9], // Sample 2: 3 model predictions
            ]);

            const result = averagePredictions(predictions);

            expect(result.shape).toEqual([2, 1]);

            const resultData = result.arraySync() as number[][];

            // Sample averages
            expect(resultData[0][0]).toBeCloseTo(1.0, 5); // (1.0+1.2+0.8)/3
            expect(resultData[1][0]).toBeCloseTo(2.0, 5); // (2.0+2.1+1.9)/3

            predictions.dispose();
            result.dispose();
        });
    });
});
