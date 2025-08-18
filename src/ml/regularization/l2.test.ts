import * as tf from '@tensorflow/tfjs';
import { L2Regularization } from './l2';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

describe('L2Regularization', () => {
    let regularization: L2Regularization;

    afterEach(() => {
        regularization?.dispose();
    });

    describe('constructor', () => {
        it('should create instance with default lambda (0)', () => {
            regularization = new L2Regularization();

            expect(regularization).toBeInstanceOf(L2Regularization);
            expect(regularization['lambda']).toBeDefined();
            expect(regularization['lambda2D']).toBeDefined();
            expect(regularization['zeros2D']).toBeDefined();
        });
    });

    describe('compute', () => {
        beforeEach(() => {
            regularization = new L2Regularization(0.1);
        });

        it('should compute L2 regularization for single feature', async () => {
            // theta = [bias, weight] = [1, 2]
            const theta = tf.tensor2d([[1], [2]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L2 term = 0.5 * lambda * ||w||^2 = 0.5 * 0.1 * (2^2) = 0.5 * 0.1 * 4 = 0.2
            expect(value[0]).toBeCloseTo(0.2, 5);

            theta.dispose();
            result.dispose();
        });

        it('should compute L2 regularization for multiple features', async () => {
            // theta = [bias, weight1, weight2] = [1, 2, 3]
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L2 term = 0.5 * lambda * ||w||^2 = 0.5 * 0.1 * (2^2 + 3^2) = 0.5 * 0.1 * 13 = 0.65
            expect(value[0]).toBeCloseTo(0.65, 5);

            theta.dispose();
            result.dispose();
        });

        it('should exclude bias term from regularization', async () => {
            // theta = [bias, weight] = [100, 2] (large bias should not affect regularization)
            const theta = tf.tensor2d([[100], [2]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L2 term should only consider weight: 0.5 * 0.1 * (2^2) = 0.2
            expect(value[0]).toBeCloseTo(0.2, 5);

            theta.dispose();
            result.dispose();
        });

        it('should return zero when lambda is zero', async () => {
            const zeroRegularization = new L2Regularization(0);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = zeroRegularization.compute(theta);
            const value = await result.data();

            expect(value[0]).toBe(0);

            theta.dispose();
            result.dispose();
            zeroRegularization.dispose();
        });

        it('should handle multiple classes (columns)', async () => {
            // theta for 2 classes: [[bias1, bias2], [weight1_1, weight1_2], [weight2_1, weight2_2]]
            const theta = tf.tensor2d([
                [1, 2], // bias terms
                [3, 4], // weights for feature 1
                [5, 6], // weights for feature 2
            ]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L2 term = 0.5 * lambda * sum(weights^2)
            // weights = [3, 4, 5, 6], sum of squares = 9 + 16 + 25 + 36 = 86
            // L2 term = 0.5 * 0.1 * 86 = 4.3
            expect(value[0]).toBeCloseTo(4.3, 5);

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

        it('should handle negative weights', async () => {
            const theta = tf.tensor2d([[1], [-2], [3]]);

            const result = regularization.compute(theta);
            const value = await result.data();

            // L2 term = 0.5 * 0.1 * ((-2)^2 + 3^2) = 0.5 * 0.1 * 13 = 0.65
            expect(value[0]).toBeCloseTo(0.65, 5);

            theta.dispose();
            result.dispose();
        });
    });

    describe('gradient', () => {
        beforeEach(() => {
            regularization = new L2Regularization(0.1);
        });

        it('should compute gradient for single feature', async () => {
            const theta = tf.tensor2d([[1], [2]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // Gradient: [0, lambda * weight] = [0, 0.1 * 2] = [0, 0.2]
            expect(values[0]).toBe(0); // bias gradient
            expect(values[1]).toBeCloseTo(0.2, 5); // weight gradient

            theta.dispose();
            result.dispose();
        });

        it('should compute gradient for multiple features', async () => {
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // Gradient: [0, lambda * weight1, lambda * weight2] = [0, 0.2, 0.3]
            expect(values[0]).toBe(0); // bias gradient
            expect(values[1]).toBeCloseTo(0.2, 5); // weight1 gradient
            expect(values[2]).toBeCloseTo(0.3, 5); // weight2 gradient

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

        it('should handle multiple classes (columns)', async () => {
            const theta = tf.tensor2d([
                [1, 2], // bias terms
                [3, 4], // weights for feature 1
                [5, 6], // weights for feature 2
            ]);

            const result = regularization.gradient(theta);
            const values = await result.array();

            // Expected gradient:
            // [[0, 0],           // bias gradients (always 0)
            //  [0.3, 0.4],       // lambda * weights for feature 1
            //  [0.5, 0.6]]       // lambda * weights for feature 2
            expect(values[0][0]).toBe(0);
            expect(values[0][1]).toBe(0);
            expect(values[1][0]).toBeCloseTo(0.3, 5);
            expect(values[1][1]).toBeCloseTo(0.4, 5);
            expect(values[2][0]).toBeCloseTo(0.5, 5);
            expect(values[2][1]).toBeCloseTo(0.6, 5);

            theta.dispose();
            result.dispose();
        });

        it('should return zero gradient when lambda is zero', async () => {
            const zeroRegularization = new L2Regularization(0);
            const theta = tf.tensor2d([[1], [2], [3]]);

            const result = zeroRegularization.gradient(theta);
            const values = await result.data();

            // All gradients should be zero when lambda = 0
            expect(values.every((v) => v === 0)).toBe(true);

            theta.dispose();
            result.dispose();
            zeroRegularization.dispose();
        });

        it('should handle negative weights correctly', async () => {
            const theta = tf.tensor2d([[1], [-2], [3]]);

            const result = regularization.gradient(theta);
            const values = await result.data();

            // Gradient: [0, lambda * (-2), lambda * 3] = [0, -0.2, 0.3]
            expect(values[0]).toBe(0);
            expect(values[1]).toBeCloseTo(-0.2, 5);
            expect(values[2]).toBeCloseTo(0.3, 5);

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
            regularization = new L2Regularization(0.1);

            expect(() => {
                regularization.dispose();
            }).not.toThrow();
            expect(regularization['lambda'].isDisposed).toBeTruthy();
            expect(regularization['lambda2D'].isDisposed).toBeTruthy();
            expect(regularization['zeros2D'].isDisposed).toBeTruthy();
        });
    });

    describe('edge cases', () => {
        beforeEach(() => {
            regularization = new L2Regularization(0.1);
        });

        it('should handle very small weights', async () => {
            const theta = tf.tensor2d([[0], [1e-10], [1e-10]]);

            const computeResult = regularization.compute(theta);
            const gradientResult = regularization.gradient(theta);

            const computeValue = await computeResult.data();
            const gradientValues = await gradientResult.data();

            expect(computeValue[0]).toBeCloseTo(0, 10);
            expect(gradientValues[1]).toBeCloseTo(1e-11, 15);
            expect(gradientValues[2]).toBeCloseTo(1e-11, 15);

            theta.dispose();
            computeResult.dispose();
            gradientResult.dispose();
        });

        it('should handle very large weights', async () => {
            const theta = tf.tensor2d([[0], [1e3], [1e3]]);

            const computeResult = regularization.compute(theta);
            const gradientResult = regularization.gradient(theta);

            const computeValue = await computeResult.data();
            const gradientValues = await gradientResult.data();

            // L2 term = 0.5 * 0.1 * (1e6 + 1e6) = 0.05 * 2e6 = 1e5
            expect(computeValue[0]).toBeCloseTo(1e5, 0);
            expect(gradientValues[1]).toBeCloseTo(1e2, 0);
            expect(gradientValues[2]).toBeCloseTo(1e2, 0);

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
            regularization = new L2Regularization(0.1);
        });

        it('should satisfy linearity: gradient(a*theta) = a*gradient(theta)', async () => {
            const theta = tf.tensor2d([[1], [2], [3]]);
            const scaledTheta = theta.mul(2) as tf.Tensor2D;

            const gradient1 = regularization.gradient(theta);
            const gradient2 = regularization.gradient(scaledTheta);
            const scaledGradient1 = gradient1.mul(2);

            const values1 = await scaledGradient1.data();
            const values2 = await gradient2.data();

            for (let i = 0; i < values1.length; i++) {
                expect(values1[i]).toBeCloseTo(values2[i], 5);
            }

            theta.dispose();
            scaledTheta.dispose();
            gradient1.dispose();
            gradient2.dispose();
            scaledGradient1.dispose();
        });

        it('should satisfy quadratic property: compute(a*theta) = a^2*compute(theta)', async () => {
            const theta = tf.tensor2d([[1], [2], [3]]);
            const scaledTheta = theta.mul(2) as tf.Tensor2D;

            const compute1 = regularization.compute(theta);
            const compute2 = regularization.compute(scaledTheta);
            const scaledCompute1 = compute1.mul(4); // a^2 = 2^2 = 4

            const value1 = await scaledCompute1.data();
            const value2 = await compute2.data();

            expect(value1[0]).toBeCloseTo(value2[0], 5);

            theta.dispose();
            scaledTheta.dispose();
            compute1.dispose();
            compute2.dispose();
            scaledCompute1.dispose();
        });

        it('should have consistent relationship between compute and gradient', async () => {
            const theta = tf.tensor2d([[0], [1], [0]]);
            const epsilon = 1e-6;

            // Finite difference approximation
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
});
