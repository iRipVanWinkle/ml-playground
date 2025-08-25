import * as tf from '@tensorflow/tfjs';
import { StochasticGD } from './stochastic';
import type { OptimizeParameters } from '../types';
import { beforeEach, describe, expect, it } from 'vitest';

describe('StochasticGD', () => {
    let optimizer: StochasticGD;

    describe('constructor', () => {
        it('should create instance with default batch size', () => {
            optimizer = new StochasticGD({
                learningRate: 0.01,
                maxIterations: 100,
                withBias: false,
            });

            expect(optimizer).toBeInstanceOf(StochasticGD);
            expect(optimizer['batchSize']).toBe(1);
        });

        it('should create instance with custom batch size', () => {
            optimizer = new StochasticGD({
                learningRate: 0.01,
                maxIterations: 100,
                batchSize: 5,
            });

            expect(optimizer['batchSize']).toBe(5);
        });
    });

    describe('optimize', () => {
        beforeEach(() => {
            optimizer = new StochasticGD({
                learningRate: 0.1,
                maxIterations: 100,
                batchSize: 2,
                withBias: false,
            });
        });

        it('should optimize with stochastic sampling', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]);
            const y = tf.tensor2d([[2], [4], [6], [8]]);

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

            // Check if optimization improved
            const finalLoss = lossFunction(X, y, result);
            const lossValue = await finalLoss.data();
            expect(lossValue[0]).toBeLessThan(10);

            X.dispose();
            y.dispose();
            result.dispose();
            finalLoss.dispose();
        });

        it('should handle different batch sizes', async () => {
            const batchSizes = [1, 2, 3];
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
            ]);
            const y = tf.tensor2d([[2], [4], [6], [8], [10]]);

            for (const batchSize of batchSizes) {
                const opt = new StochasticGD({
                    learningRate: 0.01,
                    maxIterations: 50,
                    withBias: false,
                    batchSize,
                });

                const result = await opt.optimize({
                    X,
                    y,
                    lossFunction: () => tf.scalar(0.1),
                    gradientFunction: () => tf.tensor2d([[0.01], [0.01]]),
                    initTheta: tf.zeros([X.shape[1], 1]),
                } as OptimizeParameters);

                expect(result).toBeDefined();
                expect(result.shape).toEqual([2, 1]);

                result.dispose();
            }

            X.dispose();
            y.dispose();
        });

        it('should throw error when batch size exceeds sample count', async () => {
            const opt = new StochasticGD({
                learningRate: 0.01,
                maxIterations: 10,
                batchSize: 5,
                withBias: false,
            });

            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
            ]); // Only 2 samples
            const y = tf.tensor2d([[1], [2]]);

            await expect(
                opt.optimize({
                    X,
                    y,
                    lossFunction: () => tf.scalar(1),
                    gradientFunction: () => tf.tensor2d([[0], [0]]),
                    initTheta: tf.zeros([X.shape[1], 1]),
                } as OptimizeParameters),
            ).rejects.toThrow('Batch size cannot be larger than the number of samples.');

            X.dispose();
            y.dispose();
        });

        it('should create different batches across iterations', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
            ]);
            const y = tf.tensor2d([[1], [2], [3], [4], [5]]);

            const batchSizes: number[] = [];
            let callCount = 0;

            const gradientFunction = (batchX: tf.Tensor2D) => {
                batchSizes.push(batchX.shape[0]);
                callCount++;
                if (callCount > 5) return tf.tensor2d([[0], [0]]); // Stop after a few iterations
                return tf.tensor2d([[0.01], [0.01]]);
            };

            const lossFunction = () => tf.scalar(callCount > 5 ? 1e-8 : 1);

            await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            // All batches should have the expected size
            expect(batchSizes.every((size) => size === 2)).toBe(true);

            X.dispose();
            y.dispose();
        });

        it('should handle single sample batch size (true SGD)', async () => {
            const sgdOpt = new StochasticGD({
                learningRate: 0.01,
                maxIterations: 50,
                withBias: false,
                batchSize: 1,
            });

            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
            ]);
            const y = tf.tensor2d([[2], [4], [6]]);

            const result = await sgdOpt.optimize({
                X,
                y,
                lossFunction: () => tf.scalar(0.1),
                gradientFunction: (batchX: tf.Tensor2D) => {
                    expect(batchX.shape[0]).toBe(1); // Single sample
                    return tf.tensor2d([[0.01], [0.01]]);
                },
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(result).toBeDefined();

            X.dispose();
            y.dispose();
            result.dispose();
        });
    });

    describe('createBatch', () => {
        beforeEach(() => {
            optimizer = new StochasticGD({
                learningRate: 0.01,
                maxIterations: 100,
                withBias: false,
                batchSize: 2,
            });
        });

        it('should create batches of correct size', () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]);
            const y = tf.tensor2d([[1], [2], [3], [4]]);

            const [batchX, batchY] = optimizer['createBatch'](X, y);

            expect(batchX.shape[0]).toBe(2);
            expect(batchY.shape[0]).toBe(2);
            expect(batchX.shape[1]).toBe(2); // Features
            expect(batchY.shape[1]).toBe(1); // Labels

            X.dispose();
            y.dispose();
            batchX.dispose();
            batchY.dispose();
        });

        it('should sample different indices across calls', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]);
            const y = tf.tensor2d([[1], [2], [3], [4]]);

            const batches: number[][] = [];

            // Create multiple batches and check they're different
            for (let i = 0; i < 10; i++) {
                const [batchX, batchY] = optimizer['createBatch'](X, y);
                const batchXValues = Array.from(await batchX.data());
                batches.push(batchXValues);
                batchX.dispose();
                batchY.dispose();
            }

            // At least some batches should be different
            const uniqueBatches = new Set(batches.map((b) => JSON.stringify(b)));
            expect(uniqueBatches.size).toBeGreaterThan(1);

            X.dispose();
            y.dispose();
        });

        it('should throw error for oversized batch', () => {
            const oversizedOpt = new StochasticGD({
                learningRate: 0.01,
                maxIterations: 10,
                withBias: false,
                batchSize: 5,
            });

            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
            ]);
            const y = tf.tensor2d([[1], [2]]);

            expect(() => {
                oversizedOpt['createBatch'](X, y);
            }).toThrow('Batch size cannot be larger than the number of samples.');

            X.dispose();
            y.dispose();
        });
    });

    describe('stochastic behavior', () => {
        it('should show variance in optimization path', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]);
            const y = tf.tensor2d([[2], [4], [6], [8]]);

            const results: tf.Tensor2D[] = [];

            // Run optimization multiple times
            for (let i = 0; i < 3; i++) {
                const opt = new StochasticGD({
                    learningRate: 0.01,
                    maxIterations: 20,
                    withBias: false,
                    batchSize: 2,
                });

                const result = await opt.optimize({
                    X,
                    y,
                    lossFunction: (X, y, theta) => {
                        const preds = X.matMul(theta);
                        return preds.sub(y).square().mean();
                    },
                    gradientFunction: (X, y, theta) => {
                        const preds = X.matMul(theta);
                        const errors = preds.sub(y);
                        return X.transpose().matMul(errors).div(X.shape[0]);
                    },
                    initTheta: tf.zeros([X.shape[1], 1]),
                } as OptimizeParameters);

                results.push(result);
            }

            // Results should show some variance due to stochastic sampling
            const values = results.map((r) => Array.from(r.dataSync()));

            // At least some variance should be present (not all identical)
            const allIdentical = values.every((v) => Math.abs(v[0] - values[0][0]) < 1e-10);
            expect(allIdentical).toBe(false);

            X.dispose();
            y.dispose();
            results.forEach((r) => r.dispose());
        });
    });
});
