import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { OneVsRestLogisticRegressor } from './OneVsRestLogisticRegressor';
import { BinaryCrossentropyLogits, BinaryCrossentropy } from '../../losses';
import { BatchGD } from '../../optimizers/batch';
import { L2Regularization } from '../../regularization';

describe('OneVsRestLogisticRegressor', () => {
    let model: OneVsRestLogisticRegressor;

    beforeEach(() => {
        model = new OneVsRestLogisticRegressor({
            lossFunc: new BinaryCrossentropy(),
            optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
        });
    });

    afterEach(() => {
        model.dispose?.();
    });

    describe('constructor', () => {
        it('should create instance with default parameters', () => {
            expect(model).toBeInstanceOf(OneVsRestLogisticRegressor);
            expect(model['lossFunc']).toBeInstanceOf(BinaryCrossentropy);
        });

        it('should create instance with custom parameters', () => {
            const customModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropyLogits(),
                optimizer: new BatchGD({ learningRate: 0.05, maxIterations: 100 }),
                regularization: new L2Regularization(0.01),
            });

            expect(customModel['lossFunc']).toBeInstanceOf(BinaryCrossentropyLogits);
            customModel.dispose?.();
        });
    });

    describe('train', () => {
        it('should train on simple 3-class linearly separable data', async () => {
            // Create 3 linearly separable classes
            const X = tf.tensor2d([
                // Class 0: top-left
                [-2, 2],
                [-1, 2],
                [-2, 1],
                // Class 1: top-right
                [2, 2],
                [1, 2],
                [2, 1],
                // Class 2: bottom
                [0, -2],
                [-1, -1],
                [1, -1],
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1], [2], [2], [2]]);

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
                [1, 1], // Class 0
                [1, -1], // Class 1
                [-1, 1], // Class 2
                [-1, -1], // Class 3
            ]);
            const y = tf.tensor2d([[0], [1], [2], [3]]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([3, 4]); // bias + 2 features, 4 classes

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with logits-based loss function', async () => {
            const logitsModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropyLogits(),
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
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);

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
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);

            const theta = await model.train(X, y);

            expect(theta.shape).toEqual([2, 3]); // bias + 1 feature, 3 classes

            X.dispose();
            y.dispose();
            theta.dispose();
        });

        it('should train with regularization', async () => {
            const regModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropy(),
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
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);

            const theta = await regModel.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 3]);

            X.dispose();
            y.dispose();
            theta.dispose();
            regModel.dispose?.();
        });

        it('should handle unbalanced classes', async () => {
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [1.5, 1.5],
                [1.2, 1.8], // Class 0 (4 samples)
                [-1, 1],
                [-2, 2], // Class 1 (2 samples)
                [1, -1], // Class 2 (1 sample)
            ]);
            const y = tf.tensor2d([[0], [0], [0], [0], [1], [1], [2]]);

            const theta = await model.train(X, y);

            expect(theta).toBeDefined();
            expect(theta.shape).toEqual([3, 3]);

            X.dispose();
            y.dispose();
            theta.dispose();
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
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);
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
                [1, 0, 0], // weight1 for each class
                [1, 0, 0], // weight2 for each class
            ]);

            const predictions = model.predict(X, customTheta);
            const predValues = await predictions.data();

            expect(predictions.shape).toEqual([1, 1]);
            // Class 0 should have highest score: 0 + 1*1 + 1*1 = 2
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
            const untrainedModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
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
                [2, 2], // Class 0
                [-1, 1],
                [-2, 2], // Class 1
                [0, -1],
                [0, -2], // Class 2
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);
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
            const X = tf.tensor2d([[1, 1]]);
            const y = tf.tensor2d([[0]]);
            const customTheta = tf.tensor2d([
                [1, -1, -1], // bias
                [1, 0, 0], // weight1
                [1, 0, 0], // weight2
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
            const yCorrect = tf.tensor2d([[0], [1], [2]]); // Correct labels
            const yWrong = tf.tensor2d([[1], [2], [0]]); // Wrong labels

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
            const untrainedModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropy(),
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

    describe('classesDataIterator', () => {
        it('should iterate through classes correctly', () => {
            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
                [5, 6],
            ]);
            const y = tf.tensor2d([[0], [1], [0]]);

            const iterator = model['classesDataIterator'](X, y);
            const results = Array.from(iterator);

            expect(results).toHaveLength(2); // Two unique classes: 0 and 1

            // Check that we get both classes
            const labels = results.map(([label]) => label);
            expect(labels).toContain(0);
            expect(labels).toContain(1);

            // Cleanup
            X.dispose();
            y.dispose();
            results.forEach(([, [features, labels]]) => {
                features.dispose();
                labels.dispose();
            });
        });

        it('should create correct binary labels for each class', async () => {
            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ]);
            const y = tf.tensor2d([[0], [1], [0], [2]]);

            const iterator = model['classesDataIterator'](X, y);
            const results = Array.from(iterator);

            // Find the result for class 0
            const class0Result = results.find(([label]) => label === 0);
            expect(class0Result).toBeDefined();

            const [, [, class0Labels]] = class0Result!;
            const class0LabelValues = await class0Labels.data();

            // Should be [1, 0, 1, 0] for class 0 (samples 0 and 2 are class 0)
            expect(Array.from(class0LabelValues)).toEqual([1, 0, 1, 0]);

            // Cleanup
            X.dispose();
            y.dispose();
            results.forEach(([, [features, labels]]) => {
                features.dispose();
                labels.dispose();
            });
        });

        it('should handle single class correctly', () => {
            const X = tf.tensor2d([
                [1, 2],
                [3, 4],
            ]);
            const y = tf.tensor2d([[5], [5]]); // All same class

            const iterator = model['classesDataIterator'](X, y);
            const results = Array.from(iterator);

            expect(results).toHaveLength(1); // Only one unique class
            expect(results[0][0]).toBe(5);

            // Cleanup
            X.dispose();
            y.dispose();
            results.forEach(([, [features, labels]]) => {
                features.dispose();
                labels.dispose();
            });
        });
    });

    describe('integration tests', () => {
        it('should solve iris-like 3-class classification problem', async () => {
            // Create synthetic iris-like data
            const X = tf.tensor2d([
                // Class 0: Small petals
                [1, 0.5],
                [1.2, 0.6],
                [0.8, 0.4],
                // Class 1: Medium petals
                [3, 1.5],
                [3.2, 1.6],
                [2.8, 1.4],
                // Class 2: Large petals
                [5, 2.5],
                [5.2, 2.6],
                [4.8, 2.4],
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1], [2], [2], [2]]);

            const irisModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 500 }),
            });

            await irisModel.train(X, y);

            // Test on training data (should get high accuracy)
            const predictions = irisModel.predict(X);

            const predValues = await predictions.data();
            const yValues = await y.data();

            const accuracy =
                Array.from(predValues).reduce((acc, pred, i) => {
                    return acc + (pred === yValues[i] ? 1 : 0);
                }, 0) / predValues.length;

            expect(accuracy).toBeGreaterThan(0.7); // At least 70% accuracy

            X.dispose();
            y.dispose();
            predictions.dispose();
            irisModel.dispose?.();
        });

        it('should handle many classes efficiently', async () => {
            // Test with 5 classes
            const X = tf.tensor2d([
                [1, 1],
                [2, 2], // Class 0
                [-1, 1],
                [-2, 2], // Class 1
                [1, -1],
                [2, -2], // Class 2
                [-1, -1],
                [-2, -2], // Class 3
                [0, 0],
                [0.1, 0.1], // Class 4
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2], [3], [3], [4], [4]]);

            const multiClassModel = new OneVsRestLogisticRegressor({
                lossFunc: new BinaryCrossentropy(),
                optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 300 }),
            });

            const startTime = Date.now();
            await multiClassModel.train(X, y);
            const trainTime = Date.now() - startTime;

            // Should complete training in reasonable time (< 10 seconds)
            expect(trainTime).toBeLessThan(10000);

            const predictions = multiClassModel.predict(X);
            expect(predictions.shape).toEqual([10, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
            multiClassModel.dispose?.();
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
                new OneVsRestLogisticRegressor({
                    lossFunc: new BinaryCrossentropy(),
                    optimizer: new BatchGD({ learningRate: 0.1, maxIterations: 200 }),
                }),
                new OneVsRestLogisticRegressor({
                    lossFunc: new BinaryCrossentropyLogits(),
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

        it('should demonstrate One-vs-Rest strategy', async () => {
            // Create data where One-vs-Rest makes sense
            const X = tf.tensor2d([
                [1, 1],
                [1.2, 1.1], // Class A
                [2, 2],
                [2.1, 2.2], // Class B
                [3, 1],
                [3.1, 0.9], // Class C
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);

            await model.train(X, y);

            // Each binary classifier should learn to distinguish one class from all others
            const testX = tf.tensor2d([
                [1.1, 1.05], // Should be Class A (0)
                [2.05, 2.1], // Should be Class B (1)
                [3.05, 0.95], // Should be Class C (2)
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
    });
});
