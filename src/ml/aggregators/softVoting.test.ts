import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { softVoting } from './softVoting';

describe('SoftVoting', () => {
    describe('basic functionality', () => {
        it('should average probabilities across trees correctly', () => {
            // Create a 3D tensor: [2 samples, 3 trees, 3 classes]
            const probs = tf.tensor3d([
                [
                    [0.8, 0.1, 0.1], // Tree 1 for sample 1
                    [0.7, 0.2, 0.1], // Tree 2 for sample 1
                    [0.6, 0.3, 0.1], // Tree 3 for sample 1
                ],
                [
                    [0.1, 0.8, 0.1], // Tree 1 for sample 2
                    [0.2, 0.7, 0.1], // Tree 2 for sample 2
                    [0.3, 0.6, 0.1], // Tree 3 for sample 2
                ],
            ]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([2, 3]);

            const resultData = result.arraySync() as number[][];

            // Expected averages for sample 1: [(0.8+0.7+0.6)/3, (0.1+0.2+0.3)/3, (0.1+0.1+0.1)/3]
            expect(resultData[0][0]).toBeCloseTo(0.7, 5);
            expect(resultData[0][1]).toBeCloseTo(0.2, 5);
            expect(resultData[0][2]).toBeCloseTo(0.1, 5);

            // Expected averages for sample 2: [(0.1+0.2+0.3)/3, (0.8+0.7+0.6)/3, (0.1+0.1+0.1)/3]
            expect(resultData[1][0]).toBeCloseTo(0.2, 5);
            expect(resultData[1][1]).toBeCloseTo(0.7, 5);
            expect(resultData[1][2]).toBeCloseTo(0.1, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle single tree correctly', () => {
            // Create a 3D tensor: [2 samples, 1 tree, 2 classes]
            const probs = tf.tensor3d([
                [[0.8, 0.2]], // Single tree for sample 1
                [[0.3, 0.7]], // Single tree for sample 2
            ]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([2, 2]);

            const resultData = result.arraySync() as number[][];

            // With single tree, result should be identical to input
            expect(resultData[0][0]).toBeCloseTo(0.8, 5);
            expect(resultData[0][1]).toBeCloseTo(0.2, 5);
            expect(resultData[1][0]).toBeCloseTo(0.3, 5);
            expect(resultData[1][1]).toBeCloseTo(0.7, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle binary classification correctly', () => {
            // Create a 3D tensor: [3 samples, 2 trees, 2 classes]
            const probs = tf.tensor3d([
                [
                    [0.9, 0.1], // Tree 1 for sample 1
                    [0.8, 0.2], // Tree 2 for sample 1
                ],
                [
                    [0.4, 0.6], // Tree 1 for sample 2
                    [0.3, 0.7], // Tree 2 for sample 2
                ],
                [
                    [0.6, 0.4], // Tree 1 for sample 3
                    [0.5, 0.5], // Tree 2 for sample 3
                ],
            ]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([3, 2]);

            const resultData = result.arraySync() as number[][];

            // Expected averages
            expect(resultData[0][0]).toBeCloseTo(0.85, 5); // (0.9+0.8)/2
            expect(resultData[0][1]).toBeCloseTo(0.15, 5); // (0.1+0.2)/2
            expect(resultData[1][0]).toBeCloseTo(0.35, 5); // (0.4+0.3)/2
            expect(resultData[1][1]).toBeCloseTo(0.65, 5); // (0.6+0.7)/2
            expect(resultData[2][0]).toBeCloseTo(0.55, 5); // (0.6+0.5)/2
            expect(resultData[2][1]).toBeCloseTo(0.45, 5); // (0.4+0.5)/2

            probs.dispose();
            result.dispose();
        });

        it('should handle equal probabilities correctly', () => {
            // Create a 3D tensor where all trees have equal probabilities
            const probs = tf.tensor3d([
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
            ]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(0.5, 5);
            expect(resultData[0][1]).toBeCloseTo(0.5, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle large number of trees', () => {
            // Create a 3D tensor: [1 sample, 100 trees, 3 classes]
            const numTrees = 100;
            const probsArray = Array(numTrees).fill([0.33, 0.33, 0.34]);
            const probs = tf.tensor3d([probsArray]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([1, 3]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(0.33, 2);
            expect(resultData[0][1]).toBeCloseTo(0.33, 2);
            expect(resultData[0][2]).toBeCloseTo(0.34, 2);

            probs.dispose();
            result.dispose();
        });
    });

    describe('edge cases', () => {
        it('should handle zero probabilities', () => {
            const probs = tf.tensor3d([
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([1, 3]);

            const resultData = result.arraySync() as number[][];

            // Each class gets 1/3 probability
            expect(resultData[0][0]).toBeCloseTo(1 / 3, 5);
            expect(resultData[0][1]).toBeCloseTo(1 / 3, 5);
            expect(resultData[0][2]).toBeCloseTo(1 / 3, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle very small probabilities', () => {
            const probs = tf.tensor3d([
                [
                    [0.001, 0.001, 0.998],
                    [0.002, 0.003, 0.995],
                ],
            ]);

            const result = softVoting(probs);

            expect(result.shape).toEqual([1, 3]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(0.0015, 5);
            expect(resultData[0][1]).toBeCloseTo(0.002, 5);
            expect(resultData[0][2]).toBeCloseTo(0.9965, 5);

            probs.dispose();
            result.dispose();
        });
    });

    describe('error handling', () => {
        it('should throw error for non-3D tensor', () => {
            const probs2D = tf.tensor2d([
                [0.5, 0.5],
                [0.3, 0.7],
            ]);

            expect(() => softVoting(probs2D)).toThrow('Input tensor must be 3D');

            probs2D.dispose();
        });

        it('should throw error for 1D tensor', () => {
            const probs1D = tf.tensor1d([0.5, 0.5]);

            expect(() => softVoting(probs1D)).toThrow('Input tensor must be 3D');

            probs1D.dispose();
        });

        it('should throw error for 4D tensor', () => {
            const probs4D = tf.tensor4d([[[[0.5, 0.5]]]]);

            expect(() => softVoting(probs4D)).toThrow('Input tensor must be 3D');

            probs4D.dispose();
        });
    });

    describe('memory management', () => {
        it('should not leak memory during computation', () => {
            const initialTensors = tf.memory().numTensors;

            const probs = tf.tensor3d([
                [
                    [0.7, 0.3],
                    [0.6, 0.4],
                ],
            ]);

            const result = softVoting(probs);
            result.dispose();
            probs.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);
        });
    });

    describe('performance', () => {
        it('should handle large tensors efficiently', () => {
            const numSamples = 1000;
            const numTrees = 50;
            const numClasses = 10;

            // Generate random probabilities that sum to 1 for each tree prediction
            const probsData = Array(numSamples)
                .fill(null)
                .map(() =>
                    Array(numTrees)
                        .fill(null)
                        .map(() => {
                            const raw = Array(numClasses)
                                .fill(null)
                                .map(() => Math.random());
                            const sum = raw.reduce((a, b) => a + b, 0);
                            return raw.map((x) => x / sum);
                        }),
                );

            const probs = tf.tensor3d(probsData);

            const startTime = performance.now();
            const result = softVoting(probs);
            const endTime = performance.now();

            expect(result.shape).toEqual([numSamples, numClasses]);
            expect(endTime - startTime).toBeLessThan(1000); // Should complete in less than 1 second

            // Verify that probabilities are properly averaged
            const resultData = result.arraySync() as number[][];
            for (let i = 0; i < Math.min(5, numSamples); i++) {
                const sum = resultData[i].reduce((a, b) => a + b, 0);
                expect(sum).toBeCloseTo(1.0, 3); // Probabilities should sum to 1
            }

            probs.dispose();
            result.dispose();
        });
    });
});
