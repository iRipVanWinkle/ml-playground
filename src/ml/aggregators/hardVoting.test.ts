import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { hardVoting } from './hardVoting';

describe('HardVoting', () => {
    describe('basic functionality', () => {
        it('should perform hard voting correctly for binary classification', () => {
            // Create a 3D tensor: [2 samples, 3 trees, 2 classes]
            const probs = tf.tensor3d([
                [
                    [0.8, 0.2], // Tree 1 votes for class 0
                    [0.3, 0.7], // Tree 2 votes for class 1
                    [0.9, 0.1], // Tree 3 votes for class 0
                ],
                [
                    [0.2, 0.8], // Tree 1 votes for class 1
                    [0.1, 0.9], // Tree 2 votes for class 1
                    [0.4, 0.6], // Tree 3 votes for class 1
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([2, 2]);

            const resultData = result.arraySync() as number[][];

            // Sample 1: 2 votes for class 0, 1 vote for class 1
            expect(resultData[0][0]).toBeCloseTo(2 / 3, 5); // 2 out of 3 trees voted for class 0
            expect(resultData[0][1]).toBeCloseTo(1 / 3, 5); // 1 out of 3 trees voted for class 1

            // Sample 2: 0 votes for class 0, 3 votes for class 1
            expect(resultData[1][0]).toBeCloseTo(0 / 3, 5); // 0 out of 3 trees voted for class 0
            expect(resultData[1][1]).toBeCloseTo(3 / 3, 5); // 3 out of 3 trees voted for class 1

            probs.dispose();
            result.dispose();
        });

        it('should perform hard voting correctly for multi-class classification', () => {
            // Create a 3D tensor: [2 samples, 4 trees, 3 classes]
            const probs = tf.tensor3d([
                [
                    [0.8, 0.1, 0.1], // Tree 1 votes for class 0
                    [0.2, 0.7, 0.1], // Tree 2 votes for class 1
                    [0.9, 0.05, 0.05], // Tree 3 votes for class 0
                    [0.1, 0.1, 0.8], // Tree 4 votes for class 2
                ],
                [
                    [0.3, 0.6, 0.1], // Tree 1 votes for class 1
                    [0.1, 0.8, 0.1], // Tree 2 votes for class 1
                    [0.2, 0.7, 0.1], // Tree 3 votes for class 1
                    [0.05, 0.9, 0.05], // Tree 4 votes for class 1
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([2, 3]);

            const resultData = result.arraySync() as number[][];

            // Sample 1: 2 votes for class 0, 1 vote for class 1, 1 vote for class 2
            expect(resultData[0][0]).toBeCloseTo(2 / 4, 5); // 2 out of 4 trees
            expect(resultData[0][1]).toBeCloseTo(1 / 4, 5); // 1 out of 4 trees
            expect(resultData[0][2]).toBeCloseTo(1 / 4, 5); // 1 out of 4 trees

            // Sample 2: 0 votes for class 0, 4 votes for class 1, 0 votes for class 2
            expect(resultData[1][0]).toBeCloseTo(0 / 4, 5); // 0 out of 4 trees
            expect(resultData[1][1]).toBeCloseTo(4 / 4, 5); // 4 out of 4 trees
            expect(resultData[1][2]).toBeCloseTo(0 / 4, 5); // 0 out of 4 trees

            probs.dispose();
            result.dispose();
        });

        it('should handle single tree correctly', () => {
            // Create a 3D tensor: [2 samples, 1 tree, 3 classes]
            const probs = tf.tensor3d([
                [[0.8, 0.1, 0.1]], // Single tree votes for class 0
                [[0.2, 0.7, 0.1]], // Single tree votes for class 1
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([2, 3]);

            const resultData = result.arraySync() as number[][];

            // With single tree, the winning class gets probability 1.0
            expect(resultData[0][0]).toBeCloseTo(1.0, 5); // Tree voted for class 0
            expect(resultData[0][1]).toBeCloseTo(0.0, 5);
            expect(resultData[0][2]).toBeCloseTo(0.0, 5);

            expect(resultData[1][0]).toBeCloseTo(0.0, 5);
            expect(resultData[1][1]).toBeCloseTo(1.0, 5); // Tree voted for class 1
            expect(resultData[1][2]).toBeCloseTo(0.0, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle ties in voting', () => {
            // Create a scenario where there's a tie
            const probs = tf.tensor3d([
                [
                    [0.9, 0.1], // Tree 1 votes for class 0
                    [0.1, 0.9], // Tree 2 votes for class 1
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            // In a tie, both classes get equal probability
            expect(resultData[0][0]).toBeCloseTo(0.5, 5); // 1 out of 2 trees
            expect(resultData[0][1]).toBeCloseTo(0.5, 5); // 1 out of 2 trees

            probs.dispose();
            result.dispose();
        });

        it('should handle very close probabilities', () => {
            // Test where the winning class has only slightly higher probability
            const probs = tf.tensor3d([
                [
                    [0.501, 0.499], // Tree 1 barely votes for class 0
                    [0.499, 0.501], // Tree 2 barely votes for class 1
                    [0.502, 0.498], // Tree 3 barely votes for class 0
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            // 2 votes for class 0, 1 vote for class 1
            expect(resultData[0][0]).toBeCloseTo(2 / 3, 5);
            expect(resultData[0][1]).toBeCloseTo(1 / 3, 5);

            probs.dispose();
            result.dispose();
        });
    });

    describe('edge cases', () => {
        it('should handle equal probabilities in predictions', () => {
            // When tree predictions have equal probabilities, argMax should pick the first one
            const probs = tf.tensor3d([
                [
                    [0.5, 0.5], // Tie - should pick class 0
                    [0.5, 0.5], // Tie - should pick class 0
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            // Both trees vote for class 0 (due to argMax behavior on ties)
            expect(resultData[0][0]).toBeCloseTo(1.0, 5);
            expect(resultData[0][1]).toBeCloseTo(0.0, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle extreme probabilities', () => {
            const probs = tf.tensor3d([
                [
                    [0.999, 0.001],
                    [0.001, 0.999],
                    [0.999, 0.001],
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            expect(resultData[0][0]).toBeCloseTo(2 / 3, 5);
            expect(resultData[0][1]).toBeCloseTo(1 / 3, 5);

            probs.dispose();
            result.dispose();
        });

        it('should handle large number of classes', () => {
            const numClasses = 10;
            const numTrees = 5;

            // Create probabilities where each tree votes for a different class
            const probsArray = Array(numTrees)
                .fill(null)
                .map((_, treeIdx) => {
                    const probs = Array(numClasses).fill(0.01);
                    probs[treeIdx % numClasses] = 0.91; // This class wins for this tree
                    return probs;
                });

            const probs = tf.tensor3d([probsArray]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, numClasses]);

            const resultData = result.arraySync() as number[][];

            // Each of the first 5 classes should get 1 vote
            for (let i = 0; i < 5; i++) {
                expect(resultData[0][i]).toBeCloseTo(1 / 5, 5);
            }
            // Remaining classes should get 0 votes
            for (let i = 5; i < numClasses; i++) {
                expect(resultData[0][i]).toBeCloseTo(0, 5);
            }

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

            expect(() => hardVoting(probs2D)).toThrow('Input tensor must be 3D');

            probs2D.dispose();
        });

        it('should throw error for 1D tensor', () => {
            const probs1D = tf.tensor1d([0.5, 0.5]);

            expect(() => hardVoting(probs1D)).toThrow('Input tensor must be 3D');

            probs1D.dispose();
        });

        it('should throw error for 4D tensor', () => {
            const probs4D = tf.tensor4d([[[[0.5, 0.5]]]]);

            expect(() => hardVoting(probs4D)).toThrow('Input tensor must be 3D');

            probs4D.dispose();
        });
    });

    describe('comparison with soft voting', () => {
        it('should produce different results than soft voting for skewed predictions', () => {
            // Create a scenario where hard voting and soft voting would differ
            const probs = tf.tensor3d([
                [
                    [0.6, 0.4], // Tree 1: close call, votes for class 0
                    [0.9, 0.1], // Tree 2: strong vote for class 0
                    [0.51, 0.49], // Tree 3: barely votes for class 0
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            // All three trees vote for class 0 in hard voting
            expect(resultData[0][0]).toBeCloseTo(1.0, 5);
            expect(resultData[0][1]).toBeCloseTo(0.0, 5);

            // But in soft voting, the average would be:
            // Class 0: (0.6 + 0.9 + 0.51) / 3 = 0.67
            // Class 1: (0.4 + 0.1 + 0.49) / 3 = 0.33
            // So hard voting is more decisive

            probs.dispose();
            result.dispose();
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

            const result = hardVoting(probs);
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
            const result = hardVoting(probs);
            const endTime = performance.now();

            expect(result.shape).toEqual([numSamples, numClasses]);
            expect(endTime - startTime).toBeLessThan(1000); // Should complete in less than 1 second

            // Verify that probabilities sum to 1 (votes are normalized)
            const resultData = result.arraySync() as number[][];
            for (let i = 0; i < Math.min(5, numSamples); i++) {
                const sum = resultData[i].reduce((a, b) => a + b, 0);
                expect(sum).toBeCloseTo(1.0, 3);
            }

            probs.dispose();
            result.dispose();
        });
    });

    describe('mathematical properties', () => {
        it('should ensure vote counts are normalized', () => {
            const probs = tf.tensor3d([
                [
                    [0.8, 0.2, 0.0],
                    [0.1, 0.8, 0.1],
                    [0.3, 0.3, 0.4],
                    [0.9, 0.05, 0.05],
                    [0.2, 0.3, 0.5],
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 3]);

            const resultData = result.arraySync() as number[][];

            // Sum of all probabilities should equal 1
            const sum = resultData[0].reduce((a, b) => a + b, 0);
            expect(sum).toBeCloseTo(1.0, 5);

            // Each probability should be between 0 and 1
            for (let i = 0; i < 3; i++) {
                expect(resultData[0][i]).toBeGreaterThanOrEqual(0);
                expect(resultData[0][i]).toBeLessThanOrEqual(1);
            }

            probs.dispose();
            result.dispose();
        });

        it('should handle unanimous voting correctly', () => {
            const probs = tf.tensor3d([
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.7, 0.3],
                    [0.95, 0.05],
                ],
            ]);

            const result = hardVoting(probs);

            expect(result.shape).toEqual([1, 2]);

            const resultData = result.arraySync() as number[][];

            // All trees vote for class 0
            expect(resultData[0][0]).toBeCloseTo(1.0, 5);
            expect(resultData[0][1]).toBeCloseTo(0.0, 5);

            probs.dispose();
            result.dispose();
        });
    });
});
