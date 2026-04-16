import * as tf from '@tensorflow/tfjs';
import { ElasticNetRegularization } from './ElasticNetRegularization';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { L1Regularization } from './L1Regularization';
import { L2Regularization } from './L2Regularization';

describe('ElasticNetRegularization', () => {
    let regularization: ElasticNetRegularization;

    afterEach(() => {
        regularization?.dispose();
    });

    describe('constructor', () => {
        it('should create instance with default parameters', async () => {
            regularization = new ElasticNetRegularization();

            expect(regularization).toBeInstanceOf(ElasticNetRegularization);
            expect(regularization['alpha']).toBeDefined();
            expect(regularization['l1']).toBeDefined();
            expect(regularization['l2']).toBeDefined();

            const alphaValue = await regularization['alpha'].data();

            expect(alphaValue[0]).toBe(0.5); // default alpha
        });
    });

    describe('compute', () => {
        beforeEach(() => {
            regularization = new ElasticNetRegularization(0.1, 0.5);
        });

        it('should compute ElasticNet regularization for single feature', async () => {
            // theta = [bias, weight] = [1, 2]
            const theta = tf.tensor2d([[1], [2]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L1 term = alpha * lambda * |weight| = 0.5 * 0.1 * |2| = 0.1
            // L2 term = (1-alpha) * lambda * 0.5 * weight^2 = 0.5 * 0.1 * 0.5 * 4 = 0.1
            // Total = 0.1 + 0.1 = 0.2
            expect(value[0]).toBeCloseTo(0.2, 5);

            theta.dispose();
            result.dispose();
        });

        it('should compute ElasticNet regularization for multiple features', async () => {
            // theta = [bias, weight1, weight2] = [1, 2, 3]
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L1 term = 0.5 * 0.1 * (|2| + |3|) = 0.05 * 5 = 0.25
            // L2 term = 0.5 * 0.1 * 0.5 * (4 + 9) = 0.025 * 13 = 0.325
            // Total = 0.25 + 0.325 = 0.575
            expect(value[0]).toBeCloseTo(0.575, 5);

            theta.dispose();
            result.dispose();
        });

        it('should exclude bias term from regularization', async () => {
            // theta = [bias, weight] = [100, 2] (large bias should not affect regularization)
            const theta = tf.tensor2d([[100], [2]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // Should only consider weight: same as single feature test
            expect(value[0]).toBeCloseTo(0.2, 5);

            theta.dispose();
            result.dispose();
        });

        it('should return zero when lambda is zero', async () => {
            const zeroRegularization = new ElasticNetRegularization(0, 0.5);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = zeroRegularization.compute(theta);
            const value = await result.data();

            expect(value[0]).toBe(0);

            theta.dispose();
            result.dispose();
            zeroRegularization.dispose();
        });

        it('should behave like L2 when alpha=0', async () => {
            const l2ElasticNet = new ElasticNetRegularization(0.1, 0);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = l2ElasticNet.compute(theta);
            const value = await result.data();

            // Pure L2: 0.5 * lambda * ||w||^2 = 0.5 * 0.1 * (4 + 9) = 0.65
            expect(value[0]).toBeCloseTo(0.65, 5);

            theta.dispose();
            result.dispose();
            l2ElasticNet.dispose();
        });

        it('should behave like L1 when alpha=1', async () => {
            const l1ElasticNet = new ElasticNetRegularization(0.1, 1);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = l1ElasticNet.compute(theta);
            const value = await result.data();

            // Pure L1: lambda * ||w||_1 = 0.1 * (2 + 3) = 0.5
            expect(value[0]).toBeCloseTo(0.5, 5);

            theta.dispose();
            result.dispose();
            l1ElasticNet.dispose();
        });

        it('should handle multiple classes (columns)', async () => {
            const theta = tf.tensor2d([
                [1, 2], // bias terms
                [3, 4], // weights for feature 1
                [5, 6], // weights for feature 2
            ]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L1 term = 0.5 * 0.1 * (3 + 4 + 5 + 6) = 0.05 * 18 = 0.9
            // L2 term = 0.5 * 0.1 * 0.5 * (9 + 16 + 25 + 36) = 0.025 * 86 = 2.15
            // Total = 0.9 + 2.15 = 3.05
            expect(value[0]).toBeCloseTo(3.05, 5);

            theta.dispose();
            result.dispose();
        });

        it('should handle negative weights correctly', async () => {
            const theta = tf.tensor2d([[1], [-2], [3]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L1 term = 0.5 * 0.1 * (|-2| + |3|) = 0.05 * 5 = 0.25
            // L2 term = 0.5 * 0.1 * 0.5 * (4 + 9) = 0.025 * 13 = 0.325
            // Total = 0.575
            expect(value[0]).toBeCloseTo(0.575, 5);

            theta.dispose();
            result.dispose();
        });

        it('should handle zero weights', async () => {
            const theta = tf.tensor2d([[1], [0], [0]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            expect(value[0]).toBe(0);

            theta.dispose();
            result.dispose();
        });
    });

    describe('gradient', () => {
        beforeEach(() => {
            regularization = new ElasticNetRegularization(0.1, 0.5);
        });

        it('should compute gradient for single feature', async () => {
            const theta = tf.tensor2d([[1], [2]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // L1 gradient: alpha * lambda * sign(weight) = 0.5 * 0.1 * sign(2) = 0.05
            // L2 gradient: (1-alpha) * lambda * weight = 0.5 * 0.1 * 2 = 0.1
            // Total gradient: [0, 0.05 + 0.1] = [0, 0.15]
            expect(values[0]).toBe(0); // bias gradient
            expect(values[1]).toBeCloseTo(0.15, 5); // weight gradient

            theta.dispose();
            result.dispose();
        });

        it('should compute gradient for multiple features', async () => {
            const theta = tf.tensor2d([[1], [2], [-3]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // For weight1 (2): L1 = 0.05 * 1, L2 = 0.05 * 2 → Total = 0.15
            // For weight2 (-3): L1 = 0.05 * (-1), L2 = 0.05 * (-3) → Total = -0.2
            expect(values[0]).toBe(0); // bias gradient
            expect(values[1]).toBeCloseTo(0.15, 5); // weight1 gradient
            expect(values[2]).toBeCloseTo(-0.2, 5); // weight2 gradient

            theta.dispose();
            result.dispose();
        });

        it('should set bias gradient to zero', async () => {
            const theta = tf.tensor2d([[100], [2]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // Bias gradient should always be 0, regardless of bias value
            expect(values[0]).toBe(0);

            theta.dispose();
            result.dispose();
        });

        it('should behave like L2 gradient when alpha=0', async () => {
            const l2ElasticNet = new ElasticNetRegularization(0.1, 0);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = l2ElasticNet.gradient(theta);
            const values = await result.data();

            // Pure L2 gradient: lambda * weight
            expect(values[0]).toBe(0);
            expect(values[1]).toBeCloseTo(0.2, 5); // 0.1 * 2
            expect(values[2]).toBeCloseTo(0.3, 5); // 0.1 * 3

            theta.dispose();
            result.dispose();
            l2ElasticNet.dispose();
        });

        it('should behave like L1 gradient when alpha=1', async () => {
            const l1ElasticNet = new ElasticNetRegularization(0.1, 1);
            const theta = tf.tensor2d([[1], [2], [-3]]);

            const result = l1ElasticNet.gradient(theta);
            const values = await result.data();

            // Pure L1 gradient: lambda * sign(weight)
            expect(values[0]).toBe(0);
            expect(values[1]).toBeCloseTo(0.1, 5); // 0.1 * sign(2)
            expect(values[2]).toBeCloseTo(-0.1, 5); // 0.1 * sign(-3)

            theta.dispose();
            result.dispose();
            l1ElasticNet.dispose();
        });

        it('should handle multiple classes (columns)', async () => {
            const theta = tf.tensor2d([
                [1, 2], // bias terms
                [3, -4], // weights for feature 1
                [-5, 6], // weights for feature 2
            ]);

            const result = regularization.gradient(theta);
            const values = await result.array();

            // For each weight: L1 + L2 gradient
            // bias gradients are always 0
            expect(values[0][0]).toBe(0);
            expect(values[0][1]).toBe(0);

            // weight (3,0): L1 = 0.05*1, L2 = 0.05*3 → 0.2
            expect(values[1][0]).toBeCloseTo(0.2, 5);
            // weight (-4,1): L1 = 0.05*(-1), L2 = 0.05*(-4) → -0.25
            expect(values[1][1]).toBeCloseTo(-0.25, 5);
            // weight (-5,0): L1 = 0.05*(-1), L2 = 0.05*(-5) → -0.3
            expect(values[2][0]).toBeCloseTo(-0.3, 5);
            // weight (6,1): L1 = 0.05*1, L2 = 0.05*6 → 0.35
            expect(values[2][1]).toBeCloseTo(0.35, 5);

            theta.dispose();
            result.dispose();
        });

        it('should return zero gradient when lambda is zero', async () => {
            const zeroRegularization = new ElasticNetRegularization(0, 0.5);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = zeroRegularization.gradient(theta);
            const values = await result.data();

            // All gradients should be zero when lambda = 0
            expect(values.every((v) => v === 0)).toBe(true);

            theta.dispose();
            result.dispose();
            zeroRegularization.dispose();
        });

        it('should handle zero weights', async () => {
            const theta = tf.tensor2d([[1], [0], [2]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // For zero weight: L1 = 0.05 * 0, L2 = 0.05 * 0 → 0
            expect(values[0]).toBe(0); // bias
            expect(values[1]).toBe(0); // zero weight
            expect(values[2]).toBeCloseTo(0.15, 5); // non-zero weight

            theta.dispose();
            result.dispose();
        });

        it('should maintain tensor shape', async () => {
            const theta = tf.tensor2d([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]);

            const result = regularization.gradient(theta);

            expect(result.shape).toEqual([3, 3]);
            expect(result.shape).toEqual(theta.shape);

            theta.dispose();
            result.dispose();
        });
    });

    describe('dispose', () => {
        it('should dispose all tensors without errors', () => {
            regularization = new ElasticNetRegularization(0.1, 0.5);

            expect(() => {
                regularization.dispose();
            }).not.toThrow();
            expect(regularization['alpha'].isDisposed).toBeTruthy();
            expect(regularization['alpha2D'].isDisposed).toBeTruthy();
            expect(regularization['zeros2D'].isDisposed).toBeTruthy();
        });
    });

    describe('edge cases', () => {
        beforeEach(() => {
            regularization = new ElasticNetRegularization(0.1, 0.5);
        });

        it('should handle very small weights', async () => {
            const theta = tf.tensor2d([[0], [1e-10], [-1e-10]]);

            const computeResult = regularization.compute(theta);
            const gradientResult = regularization.gradient(theta);

            const computeValue = await computeResult.data();
            const gradientValues = await gradientResult.data();

            expect(Number.isFinite(computeValue[0])).toBe(true);
            expect(gradientValues.every((v) => Number.isFinite(v))).toBe(true);

            theta.dispose();
            computeResult.dispose();
            gradientResult.dispose();
        });

        it('should handle very large weights', async () => {
            const theta = tf.tensor2d([[0], [1e6], [-1e6]]);

            const computeResult = regularization.compute(theta);
            const gradientResult = regularization.gradient(theta);

            const computeValue = await computeResult.data();
            const gradientValues = await gradientResult.data();

            expect(Number.isFinite(computeValue[0])).toBe(true);
            expect(gradientValues.every((v) => Number.isFinite(v))).toBe(true);

            theta.dispose();
            computeResult.dispose();
            gradientResult.dispose();
        });

        it('should handle single row tensor (bias only)', async () => {
            const theta = tf.tensor2d([[5]]);

            const computeResult = regularization.compute(theta);
            const gradientResult = regularization.gradient(theta);

            const computeValue = await computeResult.data();
            const gradientValues = await gradientResult.data();

            // No weights to regularize, only bias
            expect(computeValue[0]).toBe(0);
            expect(gradientValues[0]).toBe(0);

            theta.dispose();
            computeResult.dispose();
            gradientResult.dispose();
        });
    });

    describe('mathematical properties', () => {
        beforeEach(() => {
            regularization = new ElasticNetRegularization(0.1, 0.5);
        });

        it('should be convex combination of L1 and L2', async () => {
            const theta = tf.tensor2d([[0], [2], [3]]);

            // Compute pure L1 and L2
            const l1Reg = new ElasticNetRegularization(0.1, 1);
            const l2Reg = new ElasticNetRegularization(0.1, 0);

            const elasticResult = regularization.compute(theta);
            const l1Result = l1Reg.compute(theta);
            const l2Result = l2Reg.compute(theta);

            const elasticValue = await elasticResult.data();
            const l1Value = await l1Result.data();
            const l2Value = await l2Result.data();

            // ElasticNet should be: 0.5 * L1 + 0.5 * L2
            const expected = 0.5 * l1Value[0] + 0.5 * l2Value[0];
            expect(elasticValue[0]).toBeCloseTo(expected, 5);

            theta.dispose();
            elasticResult.dispose();
            l1Result.dispose();
            l2Result.dispose();
            l1Reg.dispose();
            l2Reg.dispose();
        });

        it('should be non-negative', async () => {
            const testCases = [
                [[0], [1], [-2]],
                [[0], [-5], [10]],
                [[0], [0], [0]],
                [[0], [-1], [-1]],
            ];

            for (const weights of testCases) {
                const theta = tf.tensor2d(weights);
                const result = regularization.compute(theta);
                const value = await result.data();

                expect(value[0]).toBeGreaterThanOrEqual(0);

                theta.dispose();
                result.dispose();
            }
        });

        it('should interpolate between L1 and L2 based on alpha', async () => {
            const theta = tf.tensor2d([[0], [2], [3]]);
            const alphas = [0, 0.25, 0.5, 0.75, 1];
            const results = [];

            for (const alpha of alphas) {
                const reg = new ElasticNetRegularization(0.1, alpha);
                const result = reg.compute(theta);
                const value = await result.data();
                results.push(value[0]);
                result.dispose();
                reg.dispose();
            }

            // Results should be monotonic or show smooth transition
            // between pure L2 (alpha=0) and pure L1 (alpha=1)
            expect(results[0]).toBeCloseTo(0.65, 5); // Pure L2
            expect(results[4]).toBeCloseTo(0.5, 5); // Pure L1

            // Intermediate values should be between L1 and L2
            for (let i = 1; i < 4; i++) {
                expect(results[i]).toBeGreaterThan(Math.min(results[0], results[4]));
                expect(results[i]).toBeLessThan(Math.max(results[0], results[4]));
            }

            theta.dispose();
        });

        it('should have consistent gradient-compute relationship', async () => {
            const theta = tf.tensor2d([[0], [1], [0]]);
            const epsilon = 1e-6;

            // Finite difference approximation for the second parameter
            const thetaPlus = tf.tensor2d([[0], [1 + epsilon], [0]]);
            const thetaMinus = tf.tensor2d([[0], [1 - epsilon], [0]]);

            const computePlus = regularization.compute(thetaPlus);
            const computeMinus = regularization.compute(thetaMinus);
            const gradient = regularization.gradient(theta);

            const plusValue = await computePlus.data();
            const minusValue = await computeMinus.data();
            const gradientValues = await gradient.data();

            const finiteDiff = (plusValue[0] - minusValue[0]) / (2 * epsilon);
            const analyticalGrad = gradientValues[1]; // gradient w.r.t. second parameter

            expect(analyticalGrad).toBeCloseTo(finiteDiff, 2);

            theta.dispose();
            thetaPlus.dispose();
            thetaMinus.dispose();
            computePlus.dispose();
            computeMinus.dispose();
            gradient.dispose();
        });
    });

    describe('comparison with pure regularizers', () => {
        it('should reduce to L1 when alpha=1', async () => {
            const elasticNet = new ElasticNetRegularization(0.1, 1);
            const l1Reg = new L1Regularization(0.1);

            const theta = tf.tensor2d([[0], [2], [-3]]);

            const elasticResult = elasticNet.compute(theta);
            const l1Result = l1Reg.compute(theta);

            const elasticValue = await elasticResult.data();
            const l1Value = await l1Result.data();

            expect(elasticValue[0]).toBeCloseTo(l1Value[0], 5);

            theta.dispose();
            elasticResult.dispose();
            l1Result.dispose();
            elasticNet.dispose();
            l1Reg.dispose();
        });

        it('should reduce to L2 when alpha=0', async () => {
            const elasticNet = new ElasticNetRegularization(0.1, 0);
            const l2Reg = new L2Regularization(0.1);

            const theta = tf.tensor2d([[0], [2], [-3]]);

            const elasticResult = elasticNet.compute(theta);
            const l2Result = l2Reg.compute(theta);
            elasticResult.print();
            l2Result.print();
            const elasticValue = await elasticResult.data();
            const l2Value = await l2Result.data();

            expect(elasticValue[0]).toBeCloseTo(l2Value[0], 5);

            theta.dispose();
            elasticResult.dispose();
            l2Result.dispose();
            elasticNet.dispose();
            l2Reg.dispose();
        });
    });
});
