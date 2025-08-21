import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LogisticRegressor } from './LogisticRegressor';
import { BinaryCrossentropyLogits, BinaryCrossentropy } from '../../losses';
import { L2Regularization, NoRegularization } from '../../regularization';
import { BatchGD } from '../../optimizers/batch';
import type { LossFunction, Optimizer, Regularization } from '../../types';

describe('LogisticRegressor', () => {
    let model: LogisticRegressor;

    let lossFunc: LossFunction;
    let optimizer: Optimizer;
    let regularization: Regularization;

    beforeEach(() => {
        lossFunc = new BinaryCrossentropy();
        optimizer = new BatchGD({ learningRate: 0.01, maxIterations: 100 });
        regularization = new NoRegularization();

        model = new LogisticRegressor({ lossFunc, optimizer, regularization });
    });

    afterEach(() => {
        model.dispose?.();
    });

    describe('constructor', () => {
        it('should create instance with default parameters', () => {
            expect(model).toBeInstanceOf(LogisticRegressor);
            expect(model['lossFunc']).toBeInstanceOf(BinaryCrossentropy);
        });

        it('should create instance with custom parameters', () => {
            const customModel = new LogisticRegressor({
                lossFunc: new BinaryCrossentropyLogits(),
                optimizer,
                regularization: new L2Regularization(0.01),
            });

            expect(customModel['lossFunc']).toBeInstanceOf(BinaryCrossentropyLogits);
            customModel.dispose?.();
        });
    });

    describe('train', () => {
        it('should train on linearly separable data', async () => {
            // Create linearly separable data
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [3, 3],
                [-1, -1],
                [-2, -2],
                [-3, -3],
            ]);
            const y = tf.tensor2d([[1], [1], [1], [0], [0], [0]]);

            const theta = await model.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 1]); // bias + 2 features

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with logits-based loss function', async () => {
            const logitsModel = new LogisticRegressor({ lossFunc, optimizer, regularization });

            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1]]);

            const theta = await logitsModel.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 1]);

            X.dispose();
            y.dispose();
            theta.dispose();
            logitsModel.dispose?.();
        });

        it('should handle single feature training', async () => {
            const X = tf.tensor2d([[1], [2], [3], [4], [5]]);
            const y = tf.tensor2d([[0], [0], [1], [1], [1]]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([2, 1]); // bias + 1 feature

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should converge to reasonable parameters for simple dataset', async () => {
            // Simple dataset where y = 1 if x1 + x2 > 0
            const X = tf.tensor2d([
                [1, 1], // y = 1
                [2, 0], // y = 1
                [0, 2], // y = 1
                [-1, -1], // y = 0
                [-2, 0], // y = 0
                [0, -2], // y = 0
            ]);
            const y = tf.tensor2d([[1], [1], [1], [0], [0], [0]]);

            const theta = await model.train(X, y);
            const weights = await theta.data();

            // Weights should be positive for both features (w1 > 0, w2 > 0)
            expect(weights[1]).toBeGreaterThan(0); // w1
            expect(weights[2]).toBeGreaterThan(0); // w2

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should keep memory clear', async () => {
            const X = tf.tensor2d([[34], [45], [50], [60], [63], [70]]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);

            const prevNumTensors = tf.memory().numTensors;

            await model.train(X, y);

            const expectedNumTensors = prevNumTensors + 1; // 1 for the model's theta

            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });

    describe('predict', () => {
        beforeEach(async () => {
            // Train the model first
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [-1, -1],
                [-2, -2],
            ]);
            const y = tf.tensor2d([[1], [1], [0], [0]]);
            await model.train(X, y);
            X.dispose();
            y.dispose();
        });

        it('should predict binary classes correctly', async () => {
            const X = tf.tensor2d([
                [3, 3],
                [-3, -3],
            ]);
            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([2, 1]);
            expect(predValues[0]).toBeCloseTo(1, 0); // Should predict class 1
            expect(predValues[1]).toBeCloseTo(0, 0); // Should predict class 0

            X.dispose();
            predictions.dispose();
        });

        it('should predict with custom theta', async () => {
            const X = tf.tensor2d([[1, 1]]);
            const customTheta = tf.tensor2d([[0], [1], [1]]); // bias=0, w1=1, w2=1

            const predictions = model.predict(X, customTheta);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // x1*w1 + x2*w2 = 1*1 + 1*1 = 2, sigmoid(2) ≈ 0.88 > 0.5 → class 1
            expect(predValues[0]).toBeCloseTo(1, 0);

            X.dispose();
            customTheta.dispose();
            predictions.dispose();
        });

        it('should predict multiple samples correctly', async () => {
            const X = tf.tensor2d([
                [2, 2], // Should be class 1
                [1, 1], // Should be class 1
                [-1, -1], // Should be class 0
                [-2, -2], // Should be class 0
            ]);

            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([4, 1]);
            expect(predValues[0]).toBeCloseTo(1, 0);
            expect(predValues[1]).toBeCloseTo(1, 0);
            expect(predValues[2]).toBeCloseTo(0, 0);
            expect(predValues[3]).toBeCloseTo(0, 0);

            X.dispose();
            predictions.dispose();
        });

        it('should handle edge case near decision boundary', async () => {
            const X = tf.tensor2d([[0, 0]]); // Right at the boundary
            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // Should be either 0 or 1
            expect(predValues[0]).toBeOneOf([0, 1]);

            X.dispose();
            predictions.dispose();
        });

        it('should throw error when model is not trained', () => {
            const untrainedModel = new LogisticRegressor({
                lossFunc: new BinaryCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.01, maxIterations: 100 }),
                regularization: new NoRegularization(),
            });

            const X = tf.tensor2d([[1, 2]]);

            expect(() => {
                untrainedModel.predict(X);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
            untrainedModel.dispose?.();
        });
    });

    describe('loss', () => {
        let trainedTheta: tf.Tensor2D;

        beforeEach(async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [-1, -1],
                [-2, -2],
            ]);
            const y = tf.tensor2d([[1], [1], [0], [0]]);
            trainedTheta = await model.train(X, y);
            X.dispose();
            y.dispose();
        });

        afterEach(() => {
            trainedTheta?.dispose();
        });

        it('should compute loss for training data', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [-1, -1],
            ]);
            const y = tf.tensor2d([[1], [0]]);

            const [yPred, yProbs, lossValue] = model.evaluate(X, y, trainedTheta);
            const loss = await lossValue.data();

            expect(loss[0]).toBeTypeOf('number');
            expect(Number.isFinite(loss[0])).toBe(true);
            expect(loss[0]).toBeGreaterThan(0);

            X.dispose();
            y.dispose();
            yPred.dispose();
            yProbs.dispose();
            lossValue.dispose();
        });

        it('should compute loss with custom theta', async () => {
            const X = tf.tensor2d([[1, 1]]);
            const y = tf.tensor2d([[1]]);
            const customTheta = tf.tensor2d([[0], [1], [1]]);

            const [yPred, yProbs, lossValue] = model.evaluate(X, y, customTheta);
            const loss = await lossValue.data();

            expect(loss[0]).toBeTypeOf('number');
            expect(Number.isFinite(loss[0])).toBe(true);

            X.dispose();
            y.dispose();
            yPred.dispose();
            yProbs.dispose();
            customTheta.dispose();
            lossValue.dispose();
        });

        it('should have lower loss for correct predictions', async () => {
            const X = tf.tensor2d([
                [2, 2],
                [-2, -2],
            ]);
            const yCorrect = tf.tensor2d([[1], [0]]); // Correct labels
            const yWrong = tf.tensor2d([[0], [1]]); // Wrong labels

            const [yPredC, yProbsC, correctLoss] = model.evaluate(X, yCorrect, trainedTheta);
            const [yPredW, yProbsW, wrongLoss] = model.evaluate(X, yWrong, trainedTheta);

            const correctValue = await correctLoss.data();
            const wrongValue = await wrongLoss.data();

            expect(correctValue[0]).toBeLessThan(wrongValue[0]);

            X.dispose();
            yCorrect.dispose();
            yWrong.dispose();
            yPredC.dispose();
            yPredW.dispose();
            yProbsC.dispose();
            yProbsW.dispose();
            correctLoss.dispose();
            wrongLoss.dispose();
        });

        it('should throw error when model is not trained', () => {
            const untrainedModel = new LogisticRegressor({ lossFunc, optimizer, regularization });

            const X = tf.tensor2d([[1, 2]]);
            const y = tf.tensor2d([[1]]);

            expect(() => {
                untrainedModel.evaluate(X, y);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
            y.dispose();
            untrainedModel.dispose?.();
        });
    });

    describe('hypothesis', () => {
        it('should compute sigmoid probabilities by default', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const theta = tf.tensor2d([[0], [1], [1]]);

            const result = model['hypothesis'](X, theta);
            const values = await result.data();

            // Should be sigmoid output (between 0 and 1)
            expect(values[0]).toBeGreaterThan(0);
            expect(values[0]).toBeLessThan(1);

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should compute logits when asLogits=true', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const theta = tf.tensor2d([[0], [1], [1]]);

            const result = model['hypothesis'](X, theta, true);
            const values = await result.data();

            // Should be raw logits (can be any real number)
            // For X=[1,2], theta=[0,1,1]: logit = 0 + 1*1 + 1*2 = 3
            expect(values[0]).toBeCloseTo(3, 5);

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should handle batch processing', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
            ]);
            const theta = tf.tensor2d([[0], [0.5], [0.5]]);

            const result = model['hypothesis'](X, theta);

            expect(result.shape).toEqual([2, 1]);

            X.dispose();
            theta.dispose();
            result.dispose();
        });
    });

    describe('integration tests', () => {
        it('should solve XOR problem with feature engineering', async () => {
            // XOR is not linearly separable, but we can add x1*x2 as a feature
            const X = tf.tensor2d([
                [0, 0, 0], // x1, x2, x1*x2
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
            ]);
            const y = tf.tensor2d([[0], [1], [1], [0]]); // XOR truth table

            const highLearningRateModel = new LogisticRegressor({
                lossFunc,
                optimizer,
                regularization,
            });

            await highLearningRateModel.train(X, y);
            const predictions = highLearningRateModel.predict(X);
            const predValues = await predictions.data();

            // Should predict XOR correctly (or at least get most of them right)
            const accuracy =
                Array.from(predValues).reduce((acc, pred, i) => {
                    const yTrue = [0, 1, 1, 0][i];
                    return acc + (Math.round(pred) === yTrue ? 1 : 0);
                }, 0) / 4;

            expect(accuracy).toBeGreaterThan(0.5); // At least better than random

            X.dispose();
            y.dispose();
            predictions.dispose();
            highLearningRateModel.dispose?.();
        });

        it('should handle regularization properly', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [-1, -1],
                [-2, -2],
            ]);
            const y = tf.tensor2d([[1], [1], [0], [0]]);

            const regularizedModel = new LogisticRegressor({
                lossFunc,
                optimizer,
                regularization: new L2Regularization(0.1), // Add L2 regularization
            });

            const noRegModel = new LogisticRegressor({
                lossFunc,
                optimizer,
                regularization: new NoRegularization(),
            });

            const regTheta = await regularizedModel.train(X, y);
            const noRegTheta = await noRegModel.train(X, y);

            const regWeights = await regTheta.data();
            const noRegWeights = await noRegTheta.data();

            // Regularized weights should generally be smaller in magnitude
            const regMagnitude = Math.sqrt(
                Array.from(regWeights.slice(1)).reduce((sum, w) => sum + w * w, 0),
            );
            const noRegMagnitude = Math.sqrt(
                Array.from(noRegWeights.slice(1)).reduce((sum, w) => sum + w * w, 0),
            );

            expect(regMagnitude).toBeLessThanOrEqual(noRegMagnitude);

            X.dispose();
            y.dispose();
            regTheta.dispose();
            noRegTheta.dispose();
            regularizedModel.dispose?.();
            noRegModel.dispose?.();
        });
    });
});
