import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { Entropy } from './Entropy';

describe('Entropy', () => {
    let entropy: Entropy;

    beforeEach(() => {
        entropy = new Entropy();
    });

    afterEach(() => {
        entropy.dispose();

        // Clean up any remaining tensors
        const numTensorsBefore = tf.memory().numTensors;
        if (numTensorsBefore > 0) {
            console.warn(`${numTensorsBefore} tensors remaining after test`);
        }
    });

    describe('impurity', () => {
        it('should return 0 for pure dataset (all same class)', () => {
            // All samples belong to class 0
            const yValues = [
                [1, 0, 0], // class 0
                [1, 0, 0], // class 0
                [1, 0, 0], // class 0
            ];

            const result = entropy.impurity(yValues);

            expect(result).toBeCloseTo(0, 3);
        });

        it('should return maximum entropy for evenly distributed three classes', () => {
            // Equal distribution among three classes
            const yValues = [
                [1, 0, 0], // class 0
                [1, 0, 0], // class 0
                [0, 1, 0], // class 1
                [0, 1, 0], // class 1
                [0, 0, 1], // class 2
                [0, 0, 1], // class 2
            ];

            const result = entropy.impurity(yValues);

            // For three classes with equal distribution: Entropy = -3*(1/3)*log(1/3) = -log(1/3) = log(3) ≈ 1.0986
            expect(result).toBeCloseTo(Math.log(3), 6);
        });

        it('should compute correct entropy for uneven class distribution', () => {
            // 3 samples class 0, 1 sample class 1
            const yValues = [
                [1, 0], // class 0
                [1, 0], // class 0
                [1, 0], // class 0
                [0, 1], // class 1
            ];
            const result = entropy.impurity(yValues);

            // p0 = 3/4 = 0.75, p1 = 1/4 = 0.25
            // Entropy = -(0.75*log(0.75) + 0.25*log(0.25))
            const expectedEntropy = -(0.75 * Math.log(0.75) + 0.25 * Math.log(0.25));
            expect(result).toBeCloseTo(expectedEntropy, 6);
        });

        it('should handle single sample', () => {
            const yValues = [[1, 0, 0]];

            const result = entropy.impurity(yValues);

            // Single sample should have 0 entropy (pure)
            expect(result).toBeCloseTo(0, 3);
        });

        it('should handle edge case with very small probabilities', () => {
            // Test that epsilon prevents log(0)
            const yValues = [
                [0.999, 0.001],
                [0.999, 0.001],
                [0.001, 0.999],
            ];

            const result = entropy.impurity(yValues);

            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThan(0);
        });

        it('should handle case with some zero probabilities', () => {
            // One class has zero probability
            const yValues = [
                [1, 0, 0], // only class 0
                [1, 0, 0],
                [1, 0, 0],
            ];

            const result = entropy.impurity(yValues);

            // Should be 0 since only one class is present
            expect(result).toBeCloseTo(0, 3);
        });

        it('should handle probabilistic (soft) labels', () => {
            // Test with soft labels instead of hard one-hot
            const yValues = [
                [0.8, 0.2], // mostly class 0
                [0.6, 0.4], // mostly class 0
                [0.3, 0.7], // mostly class 1
                [0.1, 0.9], // mostly class 1
            ];

            const result = entropy.impurity(yValues);

            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThan(0);
        });

        it('should be maximum for uniform distribution', () => {
            // Compare different distributions
            const uniformBinary = [
                [0.5, 0.5],
                [0.5, 0.5],
            ];

            const skewedBinary = [
                [0.9, 0.1],
                [0.9, 0.1],
            ];

            const uniformResult = entropy.impurity(uniformBinary);
            const skewedResult = entropy.impurity(skewedBinary);

            // Uniform distribution should have higher entropy
            expect(uniformResult).toBeGreaterThan(skewedResult);
        });

        it('should handle large number of classes', () => {
            // Create 10-class problem with equal distribution
            const numClasses = 10;
            const samplesPerClass = 2;

            const data: number[][] = [];
            for (let classIdx = 0; classIdx < numClasses; classIdx++) {
                for (let sample = 0; sample < samplesPerClass; sample++) {
                    const oneHot = new Array(numClasses).fill(0);
                    oneHot[classIdx] = 1;
                    data.push(oneHot);
                }
            }

            const yValues = data;

            const result = entropy.impurity(yValues);

            // Equal distribution across 10 classes: Entropy = log(10) ≈ 2.303
            expect(result).toBeCloseTo(Math.log(10), 5);
        });
    });

    describe('mathematical properties', () => {
        it('should be non-negative', () => {
            const yValues = [
                [0.1, 0.3, 0.6],
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
            ];

            const result = entropy.impurity(yValues);

            expect(result).toBeGreaterThanOrEqual(0);
        });

        it('should satisfy submodularity (concavity)', () => {
            // Test that entropy is concave: H((p+q)/2) >= (H(p) + H(q))/2
            const p1 = [
                [0.9, 0.1],
                [0.9, 0.1],
            ];
            const p2 = [
                [0.1, 0.9],
                [0.1, 0.9],
            ];
            const pMid = [
                [0.5, 0.5],
                [0.5, 0.5],
            ];

            const h1 = entropy.impurity(p1);
            const h2 = entropy.impurity(p2);
            const hMid = entropy.impurity(pMid);

            // Concavity: H(mid) >= (H(1) + H(2))/2
            expect(hMid).toBeGreaterThanOrEqual((h1 + h2) / 2 - 1e-6);
        });

        it('should have maximum value of log(n) for n classes', () => {
            const numClasses = 5;
            const data: number[][] = [];

            // Create equal distribution
            for (let i = 0; i < numClasses; i++) {
                const oneHot = new Array(numClasses).fill(0);
                oneHot[i] = 1;
                data.push(oneHot);
            }

            const yValues = data;
            const result = entropy.impurity(yValues);

            // Maximum entropy for n classes is log(n)
            expect(result).toBeCloseTo(Math.log(numClasses), 6);
        });

        it('should be symmetric with respect to class permutation', () => {
            // Test that swapping classes doesn't change entropy
            const dist1 = [
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
            ];

            const dist2 = [
                [0.3, 0.6, 0.1], // swapped first two classes
                [0.3, 0.6, 0.1],
            ];

            const entropy1 = entropy.impurity(dist1);
            const entropy2 = entropy.impurity(dist2);

            // Should be equal (within floating point precision)
            expect(Math.abs(entropy1 - entropy2)).toBeLessThan(1e-6);
        });
    });

    describe('edge cases and memory management', () => {
        it('should handle very small datasets', () => {
            const yValues = [[1, 0]];

            const result = entropy.impurity(yValues);

            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeCloseTo(0, 3); // Single class should have 0 entropy
        });

        it('should handle uniform random distribution', () => {
            // Create a uniform random distribution
            const numSamples = 100;
            const numClasses = 3;
            const data: number[][] = [];

            for (let i = 0; i < numSamples; i++) {
                const oneHot = new Array(numClasses).fill(0);
                oneHot[i % numClasses] = 1; // Cycle through classes
                data.push(oneHot);
            }

            const yValues = data;
            const result = entropy.impurity(yValues);

            // Should be close to log(3) for equal distribution
            expect(result).toBeCloseTo(Math.log(3), 3);
        });

        it('should maintain consistency with information theory', () => {
            // Test the relationship: H(X,Y) = H(X) + H(Y|X) for independent variables
            // For our case, test that entropy of combined distribution relates properly
            const yValues = [
                [0.25, 0.25, 0.25, 0.25], // uniform 4-class distribution
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ];

            const result = entropy.impurity(yValues);

            // Should equal log(4) = 2 * log(2)
            expect(result).toBeCloseTo(2 * Math.log(2), 6);
        });
    });
});
