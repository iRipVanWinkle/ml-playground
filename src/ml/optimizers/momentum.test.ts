import * as tf from '@tensorflow/tfjs';
import { MomentumGD } from './momentum';
import type { OptimizeParameters } from '../types';
import { beforeEach, describe, expect, it } from 'vitest';
import { BatchGD } from './batch';

describe('MomentumGD', () => {
    let optimizer: MomentumGD;

    describe('constructor', () => {
        it('should create instance with default parameters', () => {
            optimizer = new MomentumGD({
                learningRate: 0.01,
                maxIterations: 100,
                withBias: false,
            });

            expect(optimizer).toBeInstanceOf(MomentumGD);
            expect(optimizer['beta']).toBe(0.9);
        });

        it('should create instance with custom beta', () => {
            optimizer = new MomentumGD({
                learningRate: 0.01,
                maxIterations: 100,
                beta: 0.8,
                withBias: false,
            });

            expect(optimizer['beta']).toBe(0.8);
        });

        it('should throw error for invalid beta values', () => {
            expect(() => {
                new MomentumGD({
                    learningRate: 0.01,
                    maxIterations: 100,
                    withBias: false,
                    beta: 0,
                });
            }).toThrow('Invalid beta value: 0. It should be in the range (0, 1).');

            expect(() => {
                new MomentumGD({
                    learningRate: 0.01,
                    maxIterations: 100,
                    withBias: false,
                    beta: 1,
                });
            }).toThrow('Invalid beta value: 1. It should be in the range (0, 1).');

            expect(() => {
                new MomentumGD({
                    learningRate: 0.01,
                    maxIterations: 100,
                    withBias: false,
                    beta: -0.1,
                });
            }).toThrow('Invalid beta value: -0.1. It should be in the range (0, 1).');
        });
    });

    describe('optimize', () => {
        beforeEach(() => {
            optimizer = new MomentumGD({
                learningRate: 0.1,
                maxIterations: 100,
                withBias: false,
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

            // Check convergence
            const finalLoss = lossFunction(X, y, result);
            const lossValue = await finalLoss.data();
            expect(lossValue[0]).toBeLessThan(1);

            X.dispose();
            y.dispose();
            result.dispose();
            finalLoss.dispose();
        });

        it('should accelerate convergence compared to basic GD', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [1, 2],
            ]);
            const y = tf.tensor2d([[3], [5]]);

            const basicOptimizer = new BatchGD({
                learningRate: 0.01,
                maxIterations: 50,
                withBias: false,
            });

            const momentumOptimizer = new MomentumGD({
                learningRate: 0.01,
                maxIterations: 50,
                withBias: false,
                beta: 0.9,
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

            const basicResult = await basicOptimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const momentumResult = await momentumOptimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const basicLoss = lossFunction(X, y, basicResult);
            const momentumLoss = lossFunction(X, y, momentumResult);

            const basicLossValue = await basicLoss.data();
            const momentumLossValue = await momentumLoss.data();

            // Momentum should converge better or at least as well
            expect(momentumLossValue[0]).toBeLessThanOrEqual(basicLossValue[0]);

            X.dispose();
            y.dispose();
            basicResult.dispose();
            momentumResult.dispose();
            basicLoss.dispose();
            momentumLoss.dispose();
        });

        it('should handle different beta values', async () => {
            const betas = [0.1, 0.5, 0.9, 0.99];

            for (const beta of betas) {
                const opt = new MomentumGD({
                    learningRate: 0.01,
                    maxIterations: 50,
                    withBias: false,
                    beta,
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

        it('should maintain velocity across iterations', async () => {
            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[1]]);

            let iterationCount = 0;
            const gradientFunction = () => {
                iterationCount++;
                // Consistent gradient direction
                return tf.tensor2d([[0.1]]);
            };

            const lossFunction = () => tf.scalar(1);

            await optimizer.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            expect(iterationCount).toBeGreaterThan(1);

            X.dispose();
            y.dispose();
        });
    });

    describe('momentum-specific behavior', () => {
        it('should smooth out oscillations with consistent gradients', async () => {
            // Create a scenario where momentum helps smooth the path
            const X = tf.tensor2d([
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
            ]);
            const y = tf.tensor2d([[2], [0], [0], [-2]]);

            const momentumOpt = new MomentumGD({
                learningRate: 0.05,
                maxIterations: 100,
                withBias: false,
                beta: 0.9,
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

            const result = await momentumOpt.optimize({
                X,
                y,
                lossFunction,
                gradientFunction,
                initTheta: tf.zeros([X.shape[1], 1]),
            } as OptimizeParameters);

            const finalLoss = lossFunction(X, y, result);
            const lossValue = await finalLoss.data();

            // Should converge to a reasonable solution
            expect(lossValue[0]).toBeLessThan(1.0);
            expect(isNaN(lossValue[0])).toBe(false);

            X.dispose();
            y.dispose();
            result.dispose();
            finalLoss.dispose();
        });
    });
});
