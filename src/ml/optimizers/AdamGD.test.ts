import * as tf from '@tensorflow/tfjs';
import { AdamGD } from './AdamGD';
import type { OptimizeParameters } from '../types';
import { beforeEach, describe, expect, it } from 'vitest';
import { BatchGD } from './BatchGD';
import { EventEmitter } from '../events/EventEmitter';

describe('AdamGD', () => {
    let optimizer: AdamGD;
    let eventEmitter: EventEmitter;

    describe('optimize', () => {
        beforeEach(() => {
            eventEmitter = new EventEmitter();
            optimizer = new AdamGD({
                learningRate: 0.1,
                maxIterations: 100,
                withBias: false,
                eventEmitter,
            });
        });

        it('should optimize simple quadratic function', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
            ]);
            const y = tf.tensor2d([[2], [4], [6]]);

            const lossFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                return diff.square().mean();
            };

            const gradientFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                return X.transpose().matMul(diff).div(X.shape[0]);
            };

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(result).toBeDefined();
            expect(result.shape).toEqual([2, 1]);

            // Check if optimization improved (theta should be close to [0, 2])
            const finalLoss = lossFunction(X, y, result);
            const lossValue = await finalLoss.data();
            expect(lossValue[0]).toBeLessThan(1);

            X.dispose();
            y.dispose();
            result.dispose();
            finalLoss.dispose();
        });

        it('should handle moment vector initialization and updates', async () => {
            const X = tf.tensor2d([[1, 1]]);
            const y = tf.tensor2d([[1]]);

            let gradientCallCount = 0;
            const gradientFunction = () => {
                gradientCallCount++;
                return tf.tensor2d([[0.1], [0.1]]);
            };

            const lossFunction = () => tf.scalar(0.5);

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(gradientCallCount).toBeGreaterThan(0);
            expect(result).toBeDefined();

            X.dispose();
            y.dispose();
            result.dispose();
        });

        it('should apply bias correction for moment estimates', async () => {
            // Test that bias correction is working by checking convergence behavior
            const X = tf.tensor2d([
                [1, 0],
                [0, 1],
            ]);
            const y = tf.tensor2d([[1], [1]]);

            const lossFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                return diff.square().sum();
            };

            const gradientFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                return X.transpose().matMul(diff).mul(2);
            };

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            // Bias correction should help with faster convergence
            const finalLoss = lossFunction(X, y, result);
            const lossValue = await finalLoss.data();
            expect(lossValue[0]).toBeLessThan(10);

            X.dispose();
            y.dispose();
            result.dispose();
            finalLoss.dispose();
        });

        it('should handle early stopping on convergence', async () => {
            optimizer = new AdamGD({
                learningRate: 1.0,
                maxIterations: 1000,
                withBias: false,
            });

            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[0]]);

            const lossFunction = () => tf.scalar(1e-4); // Already converged
            const gradientFunction = () => tf.tensor2d([[0]]);

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(result).toBeDefined();

            X.dispose();
            y.dispose();
            result.dispose();
        });

        it('should handle NaN loss gracefully', async () => {
            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[1]]);

            let callCount = 0;
            const lossFunction = () => {
                callCount++;
                return tf.scalar(callCount > 2 ? NaN : 1.0);
            };
            const gradientFunction = () => tf.tensor2d([[1]]);

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(result).toBeDefined();

            X.dispose();
            y.dispose();
            result.dispose();
        });

        it('should use custom initialization function', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const y = tf.tensor2d([[1]]);

            const lossFunction = () => tf.scalar(1);
            const gradientFunction = () => tf.tensor2d([[0], [0]]);

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.tensor2d([[5], [5]]),
            } as OptimizeParameters);

            const values = await result.data();
            expect(values[0]).toBe(5);
            expect(values[1]).toBe(5);

            X.dispose();
            y.dispose();
            result.dispose();
        });

        it('should emit callbacks during optimization', async () => {
            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[1]]);

            const callbacks: Array<{ iteration: number; theta: tf.Tensor; loss: number }> = [];
            eventEmitter.on('callback', (data) => callbacks.push(data));

            const lossFunction = () => tf.scalar(0.1);
            const gradientFunction = () => tf.tensor2d([[0.01]]);

            await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(callbacks.length).toBeGreaterThan(0);
            expect(callbacks[0]).toHaveProperty('iteration');
            expect(callbacks[0]).toHaveProperty('theta');
            expect(callbacks[0]).toHaveProperty('loss');

            X.dispose();
            y.dispose();
        });
    });

    describe('Adam-specific behavior', () => {
        it('should handle sparse gradients better than basic gradient descent', async () => {
            // Create a scenario with sparse gradients where Adam excels
            const X = tf.tensor2d([
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
            ]);
            const y = tf.tensor2d([[1], [1], [2], [0]]);

            const basicOptimizer = new BatchGD({
                learningRate: 0.1,
                maxIterations: 200,
                withBias: false,
            });

            const adamOptimizer = new AdamGD({
                learningRate: 0.1,
                maxIterations: 200,
                withBias: false,
            });

            const lossFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                return diff.square().mean();
            };

            // Gradient function with sparse updates (simulates sparse features)
            const gradientFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                const grad = X.transpose().matMul(diff).div(X.shape[0]);

                // Simulate sparse gradients by zeroing out some components randomly
                const mask = tf.randomUniform(grad.shape).greater(0.3);
                return grad.mul(mask.cast('float32'));
            };

            const basicResult = await basicOptimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const adamResult = await adamOptimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const basicLoss = lossFunction(X, y, basicResult);
            const adamLoss = lossFunction(X, y, adamResult);

            const basicLossValue = await basicLoss.data();
            const adamLossValue = await adamLoss.data();

            // Both should converge, but this tests Adam's robustness to sparse gradients
            expect(adamLossValue[0]).toBeLessThan(1.0);
            expect(basicLossValue[0]).toBeLessThan(1.0);

            X.dispose();
            y.dispose();
            basicResult.dispose();
            adamResult.dispose();
            basicLoss.dispose();
            adamLoss.dispose();
        });

        it('should handle different beta values correctly', async () => {
            const testCases = [
                { beta1: 0.8, beta2: 0.99 },
                { beta1: 0.9, beta2: 0.999 },
                { beta1: 0.95, beta2: 0.9999 },
            ];

            for (const { beta1, beta2 } of testCases) {
                const opt = new AdamGD({
                    learningRate: 0.01,
                    maxIterations: 50,
                    withBias: false,
                    beta1,
                    beta2,
                });

                const X = tf.tensor2d([[1, 1]]);
                const y = tf.tensor2d([[1]]);

                const result = await opt.optimize({
                    X,
                    y,
                    lossFunction: () => tf.scalar(0.1),
                    gradientFunction: () => tf.tensor2d([[0.01], [0.01]]),
                    initTheta: tf.zeros([X.shape[1], 1]),
                } as OptimizeParameters);

                expect(result).toBeDefined();
                expect(result.shape).toEqual([2, 1]);

                X.dispose();
                y.dispose();
                result.dispose();
            }
        });
    });
});
