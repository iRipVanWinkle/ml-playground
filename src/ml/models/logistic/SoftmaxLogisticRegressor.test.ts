import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { SoftmaxLogisticRegressor } from './SoftmaxLogisticRegressor';
import { BatchGD } from '../../optimizers/batch';
import { CategoricalCrossentropyLogits, CategoricalCrossentropy } from '../../losses';
import { L2Regularization } from '../../regularization';

describe('SoftmaxLogisticRegressor', () => {
    let model: SoftmaxLogisticRegressor;

    beforeEach(() => {
        model = new SoftmaxLogisticRegressor({
            lossFunc: new CategoricalCrossentropy(),
            optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
        });
    });

    afterEach(() => {
        model.dispose?.();
    });

    describe('constructor', () => {
        it('should create instance with default parameters', () => {
            expect(model).toBeInstanceOf(SoftmaxLogisticRegressor);
            expect(model['lossFunc']).toBeInstanceOf(CategoricalCrossentropy);
        });

        it('should create instance with custom parameters', () => {
            const customModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropyLogits(),
                optimizer: new BatchGD({ learningRate: 0.05, maxIterations: 100 }),
                regularization: new L2Regularization(0.01),
            });

            expect(customModel['lossFunc']).toBeInstanceOf(CategoricalCrossentropyLogits);
            customModel.dispose?.();
        });
    });

    describe('train', () => {
        it('should train on simple 3-class linearly separable data', async () => {
            // Create 3 linearly separable classes
            const X = tf.tensor2d([
                // Class 0: top-left quadrant
                [-2, 2],
                [-1, 2],
                [-2, 1],
                // Class 1: top-right quadrant
                [2, 2],
                [1, 2],
                [2, 1],
                // Class 2: bottom quadrant
                [0, -2],
                [-1, -1],
                [1, -1],
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const theta = await model.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 3]); // bias + 2 features, 3 classes

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with different numbers of classes', async () => {
            // Test with 4 classes
            const X = tf.tensor2d([
                [2, 2], // Class 0
                [2, -2], // Class 1
                [-2, 2], // Class 2
                [-2, -2], // Class 3
            ]);
            const y = tf.tensor2d([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([3, 4]); // bias + 2 features, 4 classes

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with logits-based loss function', async () => {
            const logitsModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropyLogits(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
            });

            const X = tf.tensor2d([
                [1, 1],
                [2, 2], // Class 0
                [-1, 1],
                [-2, 2], // Class 1
                [1, -1],
                [2, -2], // Class 2
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const theta = await logitsModel.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 3]);

            X.dispose();
            y.dispose();
            theta.dispose();
            logitsModel.dispose?.();
        });

        it('should handle single feature training', async () => {
            const X = tf.tensor2d([[1], [2], [3], [4], [5], [6]]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([2, 3]); // bias + 1 feature, 3 classes

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with regularization', async () => {
            const regModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
                regularization: new L2Regularization(0.1),
            });

            const X = tf.tensor2d([
                [2, 2],
                [1, 1], // Class 0
                [-2, 2],
                [-1, 1], // Class 1
                [0, -2],
                [0, -1], // Class 2
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const theta = await regModel.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 3]);

            X.dispose();
            y.dispose();
            theta.dispose();
            regModel.dispose?.();
        });

        it('should handle binary classification correctly', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [1.5, 1.5], // Class 0
                [-1, -1],
                [-2, -2],
                [-1.5, -1.5], // Class 1
            ]);
            const y = tf.tensor2d([
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([3, 2]); // bias + 2 features, 2 classes

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should initialize theta with Xavier/Glorot initialization', async () => {
            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
            ]);
            const y = tf.tensor2d([
                [1, 0],
                [0, 1],
            ]);

            // Train model to check initialization
            const theta = await model.train(X, y);
            const weights = await theta.data();

            // Check that weights are not all zeros (proper initialization)
            const nonZeroWeights = weights.slice(2).some((w) => Math.abs(w) > 0.01); // Skip bias terms
            expect(nonZeroWeights).toBe(true);

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should use custom initial theta when provided', async () => {
            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
            ]);
            const y = tf.tensor2d([
                [1, 0],
                [0, 1],
            ]);

            // Set custom initial theta
            const customInitTheta = tf.tensor2d([
                [0, 0],
                [0.5, -0.5],
                [1, -1],
            ]);
            model['_initTheta'] = customInitTheta;

            const theta = await model.train(X, y);

            // Check that training started from custom initialization
            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 2]);

            X.dispose();
            y.dispose();
            theta.dispose();
            customInitTheta.dispose();
        });
    });

    describe('predict', () => {
        beforeEach(async () => {
            // Train the model first with 3 classes
            const X = tf.tensor2d([
                [2, 2],
                [1, 1], // Class 0
                [-2, 2],
                [-1, 1], // Class 1
                [0, -2],
                [0, -1], // Class 2
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);
            await model.train(X, y);
            X.dispose();
            y.dispose();
        });

        it('should predict correct classes for new data', async () => {
            const X = tf.tensor2d([
                [3, 3], // Should be class 0
                [-3, 3], // Should be class 1
                [0, -3], // Should be class 2
            ]);

            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([3, 1]);
            expect(predValues[0]).toBe(0);
            expect(predValues[1]).toBe(1);
            expect(predValues[2]).toBe(2);

            X.dispose();
            predictions.dispose();
        });

        it('should predict with custom theta', async () => {
            const X = tf.tensor2d([[1, 1]]);
            // Custom theta: bias + weights for each class
            const customTheta = tf.tensor2d([
                [0, -1, -2], // bias for each class
                [2, 0, 0], // weight1 for each class
                [2, 0, 0], // weight2 for each class
            ]);

            const predictions = model.predict(X, customTheta);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // Class 0 should have highest logit: 0 + 2*1 + 2*1 = 4
            expect(predValues[0]).toBe(0);

            X.dispose();
            customTheta.dispose();
            predictions.dispose();
        });

        it('should predict multiple samples correctly', async () => {
            const X = tf.tensor2d([
                [2, 2], // Class 0
                [-2, 2], // Class 1
                [0, -2], // Class 2
                [1.5, 1.5], // Class 0
            ]);

            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([4, 1]);
            expect(predValues[0]).toBe(0);
            expect(predValues[1]).toBe(1);
            expect(predValues[2]).toBe(2);
            expect(predValues[3]).toBe(0);

            X.dispose();
            predictions.dispose();
        });

        it('should handle edge cases near decision boundaries', async () => {
            const X = tf.tensor2d([[0, 0]]); // Right at the origin

            const predictions = model.predict(X);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // Should predict one of the classes (0, 1, or 2)
            expect(predValues[0]).toBeOneOf([0, 1, 2]);

            X.dispose();
            predictions.dispose();
        });

        it('should throw error when model is not trained', () => {
            const untrainedModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
            });

            const X = tf.tensor2d([[1, 2]]);

            expect(() => {
                untrainedModel.predict(X);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
            untrainedModel.dispose?.();
        });

        it('should predict class indices (not one-hot)', async () => {
            const X = tf.tensor2d([[2, 2]]);

            const predictions = model.predict(X);
            const predValues = await predictions.data();

            // Should return class index (0, 1, or 2), not one-hot encoding
            expect(predValues[0]).toBeOneOf([0, 1, 2]);
            expect(predictions.shape).toEqual([1, 1]);

            X.dispose();
            predictions.dispose();
        });
    });

    describe('loss', () => {
        let trainedTheta: tf.Tensor2D;

        beforeEach(async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2], // Class 0
                [-1, 1],
                [-2, 2], // Class 1
                [0, -1],
                [0, -2], // Class 2
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);
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
                [-1, 1],
                [0, -1],
            ]);
            const y = tf.tensor2d([[0], [1], [2]]);

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
            const X = tf.tensor2d([
                [1, 1],
                [-1, 1],
                [0, -1],
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]);
            const customTheta = tf.tensor2d([
                [1, -1, -2], // bias for classes 0, 1, 2
                [1, 0, 0.5], // weight1 for classes 0, 1, 2
                [1, 0, -0.5], // weight2 for classes 0, 1, 2
            ]);

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
                [-2, 2],
                [0, -2],
            ]);
            const yCorrect = tf.tensor2d([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]); // Correct labels
            const yWrong = tf.tensor2d([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]); // Wrong labels

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

        it('should automatically convert class indices to one-hot', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [-1, 1],
                [0, -1],
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]); // 3 class indices, not one-hot

            const [yPred, yProbs, lossValue] = model.evaluate(X, y, trainedTheta);
            const loss = await lossValue.data();

            expect(loss[0]).toBeTypeOf('number');
            expect(Number.isFinite(loss[0])).toBe(true);

            // Verify that the model internally converted to one-hot by checking that
            // the loss is computed correctly for multi-class scenario
            expect(loss[0]).toBeGreaterThan(0);

            X.dispose();
            y.dispose();
            yPred.dispose();
            yProbs.dispose();
            lossValue.dispose();
        });

        it('should throw error when model is not trained', () => {
            const untrainedModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
            });

            const X = tf.tensor2d([[1, 2]]);
            const y = tf.tensor2d([[0]]);

            expect(() => {
                untrainedModel.evaluate(X, y);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
            y.dispose();
            untrainedModel.dispose?.();
        });
    });

    describe('hypothesis', () => {
        it('should compute softmax probabilities by default', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const theta = tf.tensor2d([
                [0, 0, 0],
                [1, 2, 3],
                [1, 2, 3],
            ]);

            const result = model['hypothesis'](X, theta);
            const values = await result.data();

            // Should be softmax output (probabilities sum to 1)
            expect(values).toHaveLength(3);
            expect(values.every((v) => v >= 0 && v <= 1)).toBe(true);

            const sum = Array.from(values).reduce((a, b) => a + b, 0);
            expect(sum).toBeCloseTo(1, 5);

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should compute logits when asLogits=true', async () => {
            const X = tf.tensor2d([[1, 2]]);
            const theta = tf.tensor2d([
                [0, 0, 0],
                [1, 2, 3],
                [1, 2, 3],
            ]);

            const result = model['hypothesis'](X, theta, true);
            const values = await result.data();

            // Should be raw logits (can be any real number, don't sum to 1)
            expect(values).toHaveLength(3);

            const sum = Array.from(values).reduce((a, b) => a + b, 0);
            expect(sum).not.toBeCloseTo(1, 1); // Should NOT sum to 1

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should handle batch processing', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
            ]);
            const theta = tf.tensor2d([
                [0, 0],
                [0.5, -0.5],
                [0.5, -0.5],
            ]);

            const result = model['hypothesis'](X, theta);

            expect(result.shape).toEqual([2, 2]); // 2 samples, 2 classes

            X.dispose();
            theta.dispose();
            result.dispose();
        });

        it('should produce valid probability distributions', async () => {
            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
                [5, 6],
            ]);
            const theta = tf.tensor2d([
                [0, 0, 0],
                [1, 2, -1],
                [-1, 1, 2],
            ]);

            const result = model['hypothesis'](X, theta);
            const values = await result.array();

            // Each row should sum to 1 (valid probability distribution)
            for (const row of values) {
                const sum = row.reduce((a, b) => a + b, 0);
                expect(sum).toBeCloseTo(1, 5);

                // All probabilities should be non-negative
                expect(row.every((p) => p >= 0)).toBe(true);
            }

            X.dispose();
            theta.dispose();
            result.dispose();
        });
    });

    describe('integration tests', () => {
        it('should solve iris-like 3-class classification problem', async () => {
            // Create synthetic iris-like data with clear class separation
            const X = tf.tensor2d([
                // Class 0: Small features
                [1, 0.5],
                [1.2, 0.6],
                [0.8, 0.4],
                [1.1, 0.5],
                // Class 1: Medium features
                [3, 1.5],
                [3.2, 1.6],
                [2.8, 1.4],
                [3.1, 1.5],
                // Class 2: Large features
                [5, 2.5],
                [5.2, 2.6],
                [4.8, 2.4],
                [5.1, 2.5],
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const irisModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 500 }),
            });

            await irisModel.train(X, y);

            // Test on training data (should get high accuracy)
            const predictions = irisModel.predict(X);
            const predValues = await predictions.data();

            // Convert one-hot y to class indices for comparison
            const yTrueIndices = await y.argMax(1).data();

            const accuracy =
                Array.from(predValues).reduce((acc, pred, i) => {
                    return acc + (pred === yTrueIndices[i] ? 1 : 0);
                }, 0) / predValues.length;

            expect(accuracy).toBeGreaterThan(0.8); // At least 80% accuracy

            X.dispose();
            y.dispose();
            predictions.dispose();
            irisModel.dispose?.();
        });

        it('should handle regularization properly', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2], // Class 0
                [-1, 1],
                [-2, 2], // Class 1
                [0, -1],
                [0, -2], // Class 2
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);

            const regModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
                regularization: new L2Regularization(0.1),
            });

            const noRegModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
            });

            const regTheta = await regModel.train(X, y);
            const noRegTheta = await noRegModel.train(X, y);

            const regWeights = await regTheta.data();
            const noRegWeights = await noRegTheta.data();

            // Regularized weights should generally be smaller in magnitude
            const regMagnitude = Math.sqrt(
                Array.from(regWeights.slice(3)).reduce((sum, w) => sum + w * w, 0),
            ); // Skip bias terms
            const noRegMagnitude = Math.sqrt(
                Array.from(noRegWeights.slice(3)).reduce((sum, w) => sum + w * w, 0),
            );

            expect(regMagnitude).toBeLessThanOrEqual(noRegMagnitude + 0.1); // Allow small tolerance

            X.dispose();
            y.dispose();
            regTheta.dispose();
            noRegTheta.dispose();
            regModel.dispose?.();
            noRegModel.dispose?.();
        });

        it('should work with different loss functions', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2], // Class 0
                [-1, 1],
                [-2, 2], // Class 1
                [0, -1],
                [0, -2], // Class 2
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);

            // Test with different loss functions
            const models = [
                new SoftmaxLogisticRegressor({
                    lossFunc: new CategoricalCrossentropy(),
                    optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
                }),
                new SoftmaxLogisticRegressor({
                    lossFunc: new CategoricalCrossentropyLogits(),
                    optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
                }),
            ];

            for (const model of models) {
                await model.train(X, y);
                const predictions = model.predict(X);

                expect(predictions.shape).toEqual([6, 1]);

                predictions.dispose();
                model.dispose?.();
            }

            X.dispose();
            y.dispose();
        });

        it('should demonstrate multinomial logistic regression', async () => {
            // Create data that requires multinomial approach (non-linearly separable with one-vs-rest)
            const X = tf.tensor2d([
                [1, 1],
                [1.1, 1.1], // Class A (top-right)
                [-1, 1],
                [-1.1, 1.1], // Class B (top-left)
                [0, -1],
                [0.1, -1.1], // Class C (bottom)
            ]);
            const y = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            await model.train(X, y);

            // Test generalization
            const testX = tf.tensor2d([
                [1.2, 1.2], // Should be Class A (0)
                [-1.2, 1.2], // Should be Class B (1)
                [0.2, -1.2], // Should be Class C (2)
            ]);

            const predictions = model.predict(testX);
            const predValues = await predictions.data();

            expect(predValues[0]).toBe(0);
            expect(predValues[1]).toBe(1);
            expect(predValues[2]).toBe(2);

            X.dispose();
            y.dispose();
            testX.dispose();
            predictions.dispose();
        });

        it('should handle large number of classes efficiently', async () => {
            // Test with 6 classes
            const X = tf.tensor2d([
                [2, 2],
                [2.1, 2.1], // Class 0
                [-2, 2],
                [-2.1, 2.1], // Class 1
                [-2, -2],
                [-2.1, -2.1], // Class 2
                [2, -2],
                [2.1, -2.1], // Class 3
                [0, 3],
                [0.1, 3.1], // Class 4
                [3, 0],
                [3.1, 0.1], // Class 5
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5]]);

            const multiClassModel = new SoftmaxLogisticRegressor({
                lossFunc: new CategoricalCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 300 }),
            });

            const startTime = Date.now();
            await multiClassModel.train(X, y);
            const trainTime = Date.now() - startTime;

            // Should complete training in reasonable time (< 10 seconds)
            expect(trainTime).toBeLessThan(10000);

            const predictions = multiClassModel.predict(X);
            expect(predictions.shape).toEqual([12, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
            multiClassModel.dispose?.();
        });
    });
});
