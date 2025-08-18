import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LinearRegressor } from './LinearRegressor';
import { MeanAbsoluteError, MeanSquaredError } from '../../losses';
import { BatchGD } from '../../optimizers/batch';
import { L2Regularization, NoRegularization } from '../../regularization';
import type { LossFunction, Optimizer } from '../../types';

describe('LinearRegressor', () => {
    let model: LinearRegressor;

    let lossFunc: LossFunction;
    let optimizer: Optimizer;

    beforeEach(() => {
        lossFunc = new MeanSquaredError();
        optimizer = new BatchGD({ learningRate: 0.01, maxIterations: 100 });

        model = new LinearRegressor({ lossFunc, optimizer });
    });

    afterEach(() => {
        model.dispose?.();
    });

    describe('constructor', () => {
        it('should create instance with default parameters', () => {
            expect(model).toBeInstanceOf(LinearRegressor);
            expect(model['lossFunc']).toBeInstanceOf(MeanSquaredError);
        });

        it('should create instance with custom parameters', () => {
            const customModel = new LinearRegressor({
                lossFunc: new MeanAbsoluteError(),
                optimizer,
                regularization: new L2Regularization(0.01),
            });

            expect(customModel['lossFunc']).toBeInstanceOf(MeanAbsoluteError);
            customModel.dispose?.();
        });
    });

    describe('train', () => {
        it('should train on simple linear data', async () => {
            // y = 2x + 1
            const X = tf.tensor2d([[1], [2], [3], [4], [5]]);
            const y = tf.tensor2d([[3], [5], [7], [9], [11]]);

            const theta = await model.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([2, 1]); // bias + 1 feature

            const weights = await theta.data();
            // Should learn approximately: bias ≈ 1, weight ≈ 2
            expect(weights[0]).toBeCloseTo(1, 0); // bias
            expect(weights[1]).toBeCloseTo(2, 0); // weight

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with multiple features', async () => {
            // y = 2*x1 + 3*x2 + 1
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]);
            const y = tf.tensor2d([[6], [11], [16], [21]]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([3, 1]); // bias + 2 features

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should handle single data point', async () => {
            const X = tf.tensor2d([[5]]);
            const y = tf.tensor2d([[10]]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([2, 1]);

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with different loss functions', async () => {
            const maeModel = new LinearRegressor({
                lossFunc: new MeanAbsoluteError(),
                optimizer,
            });

            const X = tf.tensor2d([[1], [2], [3]]);
            const y = tf.tensor2d([[2], [4], [6]]);

            const theta = await maeModel.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([2, 1]);

            X.dispose();
            y.dispose();
            theta.dispose();
            maeModel.dispose?.();
        });
    });

    describe('predict', () => {
        beforeEach(async () => {
            // Train the model first: y = 2x + 1
            const X = tf.tensor2d([[1], [2], [3], [4]]);
            const y = tf.tensor2d([[3], [5], [7], [9]]);
            await model.train(X, y);
            X.dispose();
            y.dispose();
        });

        it('should predict single value correctly', async () => {
            const X = tf.tensor2d([[5]]);
            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // Should predict y = 2*5 + 1 = 11 (approximately)
            expect(predValues[0]).toBeCloseTo(11, 0);

            X.dispose();
            predictions.dispose();
        });

        it('should predict multiple values correctly', async () => {
            const X = tf.tensor2d([[6], [7], [8]]);
            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([3, 1]);
            // Should predict y = 2x + 1
            expect(predValues[0]).toBeCloseTo(13, 0); // 2*6 + 1
            expect(predValues[1]).toBeCloseTo(15, 0); // 2*7 + 1
            expect(predValues[2]).toBeCloseTo(17, 0); // 2*8 + 1

            X.dispose();
            predictions.dispose();
        });

        it('should predict with custom theta', async () => {
            const X = tf.tensor2d([[2]]);
            const customTheta = tf.tensor2d([[1], [3]]); // bias=1, weight=3

            const predictions = model.predict(X, customTheta);
            const predValues = await predictions.data();

            // Should predict y = 3*2 + 1 = 7
            expect(predValues[0]).toBeCloseTo(7, 5);

            X.dispose();
            customTheta.dispose();
            predictions.dispose();
        });

        it('should handle multi-feature prediction', async () => {
            // Retrain with multi-features
            const multiModel = new LinearRegressor({ lossFunc, optimizer });

            const trainX = tf.tensor2d([
                [1, 2],
                [2, 3],
                [3, 4],
            ]);
            const trainY = tf.tensor2d([[8], [13], [18]]); // y = 3*x1 + 2*x2 + 1
            await multiModel.train(trainX, trainY);

            const testX = tf.tensor2d([[4, 5]]);
            const predictions = multiModel.predict(testX);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // Should predict approximately y = 3*4 + 2*5 + 1 = 23
            expect(predValues[0]).toBeCloseTo(23, 0);

            trainX.dispose();
            trainY.dispose();
            testX.dispose();
            predictions.dispose();
            multiModel.dispose?.();
        });

        it('should throw error when model is not trained', () => {
            const untrainedModel = new LinearRegressor({ lossFunc, optimizer });

            const X = tf.tensor2d([[1]]);

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
            const X = tf.tensor2d([[1], [2], [3]]);
            const y = tf.tensor2d([[2], [4], [6]]);
            trainedTheta = await model.train(X, y);
            X.dispose();
            y.dispose();
        });

        afterEach(() => {
            trainedTheta?.dispose();
        });

        it('should compute loss for training data', async () => {
            const X = tf.tensor2d([[1], [2]]);
            const y = tf.tensor2d([[2], [4]]);

            const [yPred, yProbs, lossValue] = model.evaluate(X, y, trainedTheta);
            const loss = await lossValue.data();

            expect(loss[0]).toBeTypeOf('number');
            expect(Number.isFinite(loss[0])).toBe(true);
            expect(loss[0]).toBeGreaterThanOrEqual(0);

            X.dispose();
            y.dispose();
            yPred.dispose();
            yProbs.dispose();
            lossValue.dispose();
        });

        it('should compute loss with custom theta', async () => {
            const X = tf.tensor2d([[1], [2]]);
            const y = tf.tensor2d([[2], [4]]);
            const customTheta = tf.tensor2d([[0], [2]]); // Perfect fit: y = 2x

            const [yPred, yProbs, lossValue] = model.evaluate(X, y, customTheta);
            const loss = await lossValue.data();

            expect(loss[0]).toBeCloseTo(0, 5); // Should be very close to 0

            X.dispose();
            y.dispose();
            yPred.dispose();
            yProbs.dispose();
            customTheta.dispose();
            lossValue.dispose();
        });

        it('should have lower loss for better predictions', async () => {
            const X = tf.tensor2d([[1], [2]]);
            const yCorrect = tf.tensor2d([[2], [4]]); // Matches training pattern
            const yWrong = tf.tensor2d([[10], [20]]); // Wrong values

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
            const untrainedModel = new LinearRegressor({ lossFunc, optimizer });

            const X = tf.tensor2d([[1]]);
            const y = tf.tensor2d([[2]]);

            expect(() => {
                untrainedModel.evaluate(X, y);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
            y.dispose();
            untrainedModel.dispose?.();
        });
    });

    describe('hypothesis', () => {
        it('should compute linear hypothesis correctly', async () => {
            const X = tf.tensor2d([[1], [2]]);
            const theta = tf.tensor2d([[1], [2]]); // bias=1, weight=2

            const result = model['hypothesis'](X, theta);
            const values = await result.data();

            // For X=[1], result = 1 + 2*1 = 3
            // For X=[2], result = 1 + 2*2 = 5
            expect(values[0]).toBeCloseTo(3, 5);
            expect(values[1]).toBeCloseTo(5, 5);

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should handle multi-feature hypothesis', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const theta = tf.tensor2d([[1], [2], [3]]); // bias=1, w1=2, w2=3

            const result = model['hypothesis'](X, theta);
            const values = await result.data();

            // Result = 1 + 2*1 + 3*2 = 9
            expect(values[0]).toBeCloseTo(9, 5);

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should handle batch processing', async () => {
            const X = tf.tensor2d([[1], [2], [3]]);
            const theta = tf.tensor2d([[0], [1]]); // bias=0, weight=1

            const result = model['hypothesis'](X, theta);

            expect(result.shape).toEqual([3, 1]);

            X.dispose();
            theta.dispose();
            result.dispose();
        });
    });

    describe('integration tests', () => {
        it('should solve polynomial regression with feature engineering', async () => {
            // Quadratic relationship: y = x^2, engineered as features [x, x^2]
            const X = tf.tensor2d([
                [1, 1], // x=1, x^2=1
                [2, 4], // x=2, x^2=4
                [3, 9], // x=3, x^2=9
                [4, 16], // x=4, x^2=16
            ]);
            const y = tf.tensor2d([[1], [4], [9], [16]]);

            const polyModel = new LinearRegressor({
                lossFunc,
                optimizer: new BatchGD({ learningRate: 0.001, maxIterations: 1000 }), // Lower learning rate for stability
            });

            await polyModel.train(X, y);

            // Test prediction for x=5 (should predict 25)
            const testX = tf.tensor2d([[5, 25]]);
            const prediction = polyModel.predict(testX);
            const predValue = await prediction.data();

            expect(predValue[0]).toBeCloseTo(25, 0);

            X.dispose();
            y.dispose();
            testX.dispose();
            prediction.dispose();
            polyModel.dispose?.();
        });

        it('should handle regularization properly', async () => {
            const X = tf.tensor2d([[1], [2], [3], [4]]);
            const y = tf.tensor2d([[2], [4], [6], [8]]);

            const l2Model = new LinearRegressor({
                lossFunc,
                optimizer,
                regularization: new L2Regularization(0.1),
            });

            const noRegModel = new LinearRegressor({
                lossFunc,
                optimizer,
                regularization: new NoRegularization(),
            });

            const l2Theta = await l2Model.train(X, y);
            const noRegTheta = await noRegModel.train(X, y);

            const l2Weights = await l2Theta.data();
            const noRegWeights = await noRegTheta.data();

            // L2 regularized weights should generally be smaller in magnitude
            const l2Magnitude = Math.abs(l2Weights[1]); // weight (excluding bias)
            const noRegMagnitude = Math.abs(noRegWeights[1]);

            expect(l2Magnitude).toBeLessThanOrEqual(noRegMagnitude + 0.1); // Allow small tolerance

            X.dispose();
            y.dispose();
            l2Theta.dispose();
            noRegTheta.dispose();
            l2Model.dispose?.();
            noRegModel.dispose?.();
        });

        it('should handle noisy data', async () => {
            // Linear relationship with noise
            const X = tf.tensor2d([[1], [2], [3], [4], [5]]);
            const y = tf.tensor2d([[2.1], [3.9], [6.2], [7.8], [10.1]]); // y ≈ 2x with noise

            const robustModel = new LinearRegressor({
                lossFunc,
                optimizer: new BatchGD({ learningRate: 0.01, maxIterations: 500 }), // More iterations for convergence
            });

            const theta = await robustModel.train(X, y);
            const weights = await theta.data();

            // Should still learn approximately: weight ≈ 2
            expect(weights[1]).toBeCloseTo(2, 0);

            X.dispose();
            y.dispose();
            theta.dispose();
            robustModel.dispose?.();
        });

        it('should converge faster with normalized features', async () => {
            // Test with large feature values
            const X = tf.tensor2d([[100], [200], [300], [400]]);
            const y = tf.tensor2d([[200], [400], [600], [800]]);

            const normalizedX = tf.tensor2d([[1], [2], [3], [4]]); // Normalized version
            const normalizedY = tf.tensor2d([[2], [4], [6], [8]]);

            const originalModel = new LinearRegressor({
                lossFunc,
                optimizer: new BatchGD({ learningRate: 0.000001, maxIterations: 100 }), // Very small LR needed for large features
            });

            const normalizedModel = new LinearRegressor({
                lossFunc,
                optimizer: new BatchGD({ learningRate: 0.01, maxIterations: 100 }), // Normal LR for normalized features
            });

            const originalTheta = await originalModel.train(X, y);
            const normalizedTheta = await normalizedModel.train(normalizedX, normalizedY);

            // Both should learn the relationship, but normalized should be more stable
            const [yPred, yProbs, originalLoss] = originalModel.evaluate(X, y, originalTheta);
            const [yPredN, yProbsN, normalizedLoss] = normalizedModel.evaluate(
                normalizedX,
                normalizedY,
                normalizedTheta,
            );

            const originalLossValue = await originalLoss.data();
            const normalizedLossValue = await normalizedLoss.data();

            // Normalized model should achieve better (lower) loss
            expect(normalizedLossValue[0]).toBeLessThanOrEqual(originalLossValue[0]);

            X.dispose();
            y.dispose();
            yPred.dispose();
            yPredN.dispose();
            yProbs.dispose();
            yProbsN.dispose();
            normalizedX.dispose();
            normalizedY.dispose();
            originalTheta.dispose();
            normalizedTheta.dispose();
            originalLoss.dispose();
            normalizedLoss.dispose();
            originalModel.dispose?.();
            normalizedModel.dispose?.();
        });
    });
});
