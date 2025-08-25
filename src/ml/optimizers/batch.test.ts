import * as tf from '@tensorflow/tfjs';
import { BatchGD } from './batch';
import type { OptimizeParameters } from '../types';
import { beforeEach, describe, expect, it } from 'vitest';
import { LearningRate } from '../LearningRate';
import { EventEmitter } from '../events/EventEmitter';

describe('BatchGD', () => {
    let optimizer: BatchGD;

    describe('constructor', () => {
        it('should create instance with required parameters', () => {
            optimizer = new BatchGD({
                learningRate: 0.05,
                maxIterations: 200,
            });

            expect(optimizer).toBeInstanceOf(BatchGD);

            expect(optimizer['learningRate']).toBeInstanceOf(LearningRate);
            expect(optimizer['tolerance']).toBe(1e-6);
            expect(optimizer['maxIterations']).toBe(200);
        });
    });

    describe('optimize', () => {
        beforeEach(() => {
            optimizer = new BatchGD({
                learningRate: 0.1,
                maxIterations: 100,
                withBias: false,
            });
        });

        it('should optimize simple linear regression', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]);
            const y = tf.tensor2d([[3], [5], [7], [9]]);

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

            // Check if we found good parameters (should be close to [1, 2])
            const resultValues = await result.data();
            expect(Math.abs(resultValues[0] - 1)).toBeLessThan(0.5); // bias
            expect(Math.abs(resultValues[1] - 2)).toBeLessThan(0.5); // slope

            X.dispose();
            y.dispose();
            result.dispose();
        });

        it('should handle early stopping when tolerance is met', async () => {
            optimizer = new BatchGD({
                learningRate: 1.0,
                maxIterations: 1000,
                tolerance: 1e-3,
            });

            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[0]]);

            const lossFunction = () => tf.scalar(1e-4); // Already below tolerance
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

        it('should handle NaN loss and stop optimization', async () => {
            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[1]]);

            let callCount = 0;
            const lossFunction = () => {
                callCount++;
                return tf.scalar(callCount > 3 ? NaN : 1.0);
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

        it('should use entire dataset for gradient computation', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]);
            const y = tf.tensor2d([[2], [4], [6], [8]]);

            let gradientCallCount = 0;
            const gradientFunction = (batchX: tf.Tensor2D, batchY: tf.Tensor2D) => {
                gradientCallCount++;
                // Verify it uses the full dataset
                expect(batchX.shape[0]).toBe(4); // All 4 samples
                expect(batchY.shape[0]).toBe(4);
                return tf.tensor2d([[0.01], [0.01]]);
            };

            const lossFunction = () => tf.scalar(0.1);

            await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(gradientCallCount).toBeGreaterThan(0);

            X.dispose();
            y.dispose();
        });

        it('should decrease loss over iterations', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
            ]);
            const y = tf.tensor2d([[3], [5]]);

            const losses: number[] = [];

            const lossFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                const loss = diff.square().mean();
                losses.push((loss.dataSync() as Float32Array)[0]);
                return loss;
            };

            const gradientFunction = (X: tf.Tensor2D, y: tf.Tensor2D, theta: tf.Tensor2D) => {
                const predictions = X.matMul(theta);
                const diff = predictions.sub(y);
                return X.transpose().matMul(diff).div(X.shape[0]);
            };

            await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            // Loss should generally decrease
            expect(losses.length).toBeGreaterThan(1);
            expect(losses[losses.length - 1]).toBeLessThan(losses[0]);

            X.dispose();
            y.dispose();
        });

        it('should emit callback events during optimization', async () => {
            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[1]]);
            const eventEmitter = new EventEmitter();
            optimizer = new BatchGD({
                learningRate: 0.05,
                maxIterations: 200,
                eventEmitter,
            });

            const callbacks: unknown[] = [];
            eventEmitter.on('callback', (data) => callbacks.push(data));

            await optimizer.optimize({
                X,
                y,
                lossFunction: () => tf.scalar(0.1),
                gradientFunction: () => tf.tensor2d([[0.01]]),
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(callbacks.length).toBeGreaterThan(0);
            expect(callbacks[0]).toHaveProperty('iteration');
            expect(callbacks[0]).toHaveProperty('theta');
            expect(callbacks[0]).toHaveProperty('loss');
            expect(callbacks[0]).toHaveProperty('threadId');

            X.dispose();
            y.dispose();
        });

        it('should handle different learning rates', async () => {
            const learningRates = [0.001, 0.01, 0.1];

            for (const lr of learningRates) {
                const opt = new BatchGD({
                    learningRate: lr,
                    maxIterations: 50,
                    withBias: false,
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

        it('should use custom initialization function when provided', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const y = tf.tensor2d([[1]]);

            const result = await optimizer.optimize({
                X,
                y,
                lossFunction: () => tf.scalar(1),
                gradientFunction: () => tf.tensor2d([[0], [0]]), // No update
                initTheta: tf.tensor2d([[10], [20]]),
            } as OptimizeParameters);

            const values = await result.data();
            expect(values[0]).toBe(10);
            expect(values[1]).toBe(20);

            X.dispose();
            y.dispose();
            result.dispose();
        });

        it('should handle convergence with different tolerances', async () => {
            const tolerances = [1e-3, 1e-6, 1e-9];

            for (const tolerance of tolerances) {
                const opt = new BatchGD({
                    learningRate: 0.1,
                    maxIterations: 1000,
                    tolerance,
                });

                const X = tf.tensor2d([[1]]);
                const y = tf.tensor2d([[0]]);

                let iterationCount = 0;
                const lossFunction = () => {
                    iterationCount++;
                    return tf.scalar(tolerance / 10); // Below tolerance
                };

                await opt.optimize({
                    X,
                    y,
                    lossFunction,
                    gradientFunction: () => tf.tensor2d([[0]]),
                    initTheta: tf.zeros([X.shape[1], 1]),
                } as OptimizeParameters);

                // Should stop early due to tolerance
                expect(iterationCount).toBeLessThan(1000);

                X.dispose();
                y.dispose();
            }
        });
    });

    describe('batch gradient descent properties', () => {
        it('should be deterministic with same inputs', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
            ]);
            const y = tf.tensor2d([[2], [4]]);

            const opt1 = new BatchGD({
                learningRate: 0.01,
                maxIterations: 50,
                withBias: false,
            });

            const opt2 = new BatchGD({
                learningRate: 0.01,
                maxIterations: 50,
                withBias: false,
            });

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

            const result1 = await opt1.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const result2 = await opt2.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const values1 = await result1.data();
            const values2 = await result2.data();

            // Results should be identical (deterministic)
            for (let i = 0; i < values1.length; i++) {
                expect(Math.abs(values1[i] - values2[i])).toBeLessThan(1e-10);
            }

            X.dispose();
            y.dispose();
            result1.dispose();
            result2.dispose();
        });

        it('should converge to global minimum for convex functions', async () => {
            // Test with a simple convex quadratic function
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
            ]);
            const y = tf.tensor2d([[4], [6], [8]]);

            const opt = new BatchGD({
                learningRate: 0.1,
                maxIterations: 200,
                tolerance: 1e-8,
                withBias: false,
            });

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

            const result = await opt.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            // Should converge close to the true parameters [2, 2]
            const values = await result.data();
            expect(Math.abs(values[0] - 2)).toBeLessThan(0.1); // bias
            expect(Math.abs(values[1] - 2)).toBeLessThan(0.1); // slope

            X.dispose();
            y.dispose();
            result.dispose();
        });
    });
});
