import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { QuadraticNaiveBayes } from './QuadraticNaiveBayes';

describe('QuadraticNaiveBayes', () => {
    let model: QuadraticNaiveBayes;

    beforeEach(() => {
        model = new QuadraticNaiveBayes({ regularization: 1e-9 });
    });

    afterEach(() => {
        model.dispose?.();
    });

    describe('train', () => {
        it('should train on simple 2-class dataset', async () => {
            // Class 0: points around (0, 0)
            // Class 1: points around (5, 5)
            const X = tf.tensor2d([
                [0, 0],
                [1, 1],
                [-1, -1],
                [5, 5],
                [6, 6],
                [4, 4],
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);

            const params = await model.train(X, y);

            expect(params.type).toBe('quadratic');
            expect(params.classes).toEqual([0, 1]);
            expect(params.classMeans.length).toBe(2);
            expect(params.classCovariances.length).toBe(2);
            expect(params.classCovariancesInverse.length).toBe(2);
            expect(params.classPriors.length).toBe(2);

            // Check priors (should be 0.5 for balanced dataset)
            expect(params.classPriors[0]).toBeCloseTo(0.5, 5);
            expect(params.classPriors[1]).toBeCloseTo(0.5, 5);

            X.dispose();
            y.dispose();
        });

        it('should train on 3-class dataset', async () => {
            // Three distinct clusters
            const X = tf.tensor2d([
                [0, 0],
                [1, 0],
                [0, 1], // Class 0
                [5, 5],
                [6, 5],
                [5, 6], // Class 1
                [-5, -5],
                [-4, -5],
                [-5, -4], // Class 2
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1], [2], [2], [2]]);

            const params = await model.train(X, y);

            expect(params.classes).toEqual([0, 1, 2]);
            expect(params.classMeans.length).toBe(3);
            expect(params.classCovariances.length).toBe(3);

            // Check priors (should be ~0.33 for balanced dataset)
            expect(params.classPriors[0]).toBeCloseTo(1 / 3, 5);
            expect(params.classPriors[1]).toBeCloseTo(1 / 3, 5);
            expect(params.classPriors[2]).toBeCloseTo(1 / 3, 5);

            X.dispose();
            y.dispose();
        });

        it('should compute correct means for each class', async () => {
            const X = tf.tensor2d([
                [1, 2],
                [2, 3],
                [3, 4], // Class 0: mean should be [2, 3]
                [10, 20],
                [12, 22], // Class 1: mean should be [11, 21]
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1]]);

            const params = await model.train(X, y);

            // Class 0 mean: (1+2+3)/3 = 2, (2+3+4)/3 = 3
            expect(params.classMeans[0][0]).toBeCloseTo(2, 5);
            expect(params.classMeans[0][1]).toBeCloseTo(3, 5);

            // Class 1 mean: (10+12)/2 = 11, (20+22)/2 = 21
            expect(params.classMeans[1][0]).toBeCloseTo(11, 5);
            expect(params.classMeans[1][1]).toBeCloseTo(21, 5);

            X.dispose();
            y.dispose();
        });

        it('should compute covariance matrices correctly', async () => {
            // Simple dataset with known covariance
            const X = tf.tensor2d([
                [0, 0],
                [2, 2], // Class 0
                [10, 10],
                [12, 12], // Class 1
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1]]);

            const params = await model.train(X, y);

            // Check that covariance matrices are square and positive definite
            expect(params.classCovariances[0].length).toBe(2);
            expect(params.classCovariances[0][0].length).toBe(2);
            expect(params.classCovariances[1].length).toBe(2);
            expect(params.classCovariances[1][0].length).toBe(2);

            // Diagonal elements should be positive
            expect(params.classCovariances[0][0][0]).toBeGreaterThan(0);
            expect(params.classCovariances[0][1][1]).toBeGreaterThan(0);
            expect(params.classCovariances[1][0][0]).toBeGreaterThan(0);
            expect(params.classCovariances[1][1][1]).toBeGreaterThan(0);

            X.dispose();
            y.dispose();
        });

        it('should handle correlated features', async () => {
            // Features with positive correlation
            const X = tf.tensor2d([
                [1, 1],
                [2, 2],
                [3, 3], // Class 0: highly correlated features
                [10, 5],
                [11, 6],
                [12, 7], // Class 1: moderately correlated
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);

            const params = await model.train(X, y);

            // Check off-diagonal elements (covariance between features)
            const cov0_01 = params.classCovariances[0][0][1];
            const cov0_10 = params.classCovariances[0][1][0];

            // Covariance matrix should be symmetric
            expect(cov0_01).toBeCloseTo(cov0_10, 5);

            // For positively correlated features, off-diagonal should be positive
            expect(cov0_01).toBeGreaterThan(0);

            X.dispose();
            y.dispose();
        });

        it('should add regularization to prevent singular matrices', async () => {
            const customModel = new QuadraticNaiveBayes({
                regularization: 1e-3,
            });

            // Collinear data (potentially singular covariance)
            const X = tf.tensor2d([
                [1, 2],
                [2, 4],
                [3, 6], // x2 = 2*x1 (perfect collinearity)
                [10, 20],
                [11, 22],
                [12, 24],
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);

            const params = await customModel.train(X, y);

            // Model should train successfully despite collinearity
            expect(params).toBeDefined();
            expect(params.classCovariancesInverse[0]).toBeDefined();

            X.dispose();
            y.dispose();
            customModel.dispose?.();
        });

        it('should handle imbalanced dataset correctly', async () => {
            const X = tf.tensor2d([
                [0, 0],
                [1, 1],
                [0.5, 0.5],
                [0.3, 0.3], // Class 0: 4 samples
                [10, 10], // Class 1: 1 sample
            ]);
            const y = tf.tensor2d([[0], [0], [0], [0], [1]]);

            const params = await model.train(X, y);

            // Priors should reflect the imbalance
            expect(params.classPriors[0]).toBeCloseTo(0.8, 5); // 4/5
            expect(params.classPriors[1]).toBeCloseTo(0.2, 5); // 1/5

            X.dispose();
            y.dispose();
        });
    });

    describe('predict', () => {
        beforeEach(async () => {
            // Train the model first with clear separation
            const X = tf.tensor2d([
                [0, 0],
                [1, 1],
                [-1, -1],
                [10, 10],
                [11, 11],
                [9, 9],
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);
            await model.train(X, y);
            X.dispose();
            y.dispose();
        });

        it('should predict classes correctly for well-separated data', async () => {
            const X = tf.tensor2d([
                [0.5, 0.5], // Should be class 0
                [10.5, 10.5], // Should be class 1
            ]);
            const predictions = model.predict(X);
            const predValues = await predictions.array();

            expect(predictions.shape).toEqual([2, 1]);
            expect(predValues[0][0]).toBe(0);
            expect(predValues[1][0]).toBe(1);

            X.dispose();
            predictions.dispose();
        });

        it('should predict with custom params', async () => {
            const X = tf.tensor2d([[1, 1]]);
            const customParams = {
                type: 'quadratic' as const,
                classes: [0, 1],
                classMeans: [
                    [0, 0],
                    [5, 5],
                ],
                classCovariances: [
                    [
                        [1, 0],
                        [0, 1],
                    ],
                    [
                        [1, 0],
                        [0, 1],
                    ],
                ],
                classCovariancesInverse: [
                    [
                        [1, 0],
                        [0, 1],
                    ],
                    [
                        [1, 0],
                        [0, 1],
                    ],
                ],
                classCovariancesDeterminant: [1, 1],
                classPriors: [0.5, 0.5],
            };

            const predictions = model.predict(X, customParams);
            const predValues = await predictions.array();

            // [1,1] is closer to [0,0] than [5,5]
            expect(predValues[0][0]).toBe(0);

            X.dispose();
            predictions.dispose();
        });

        it('should predict multiple samples correctly', async () => {
            const X = tf.tensor2d([
                [0, 0],
                [1, 1],
                [10, 10],
                [11, 11],
            ]);

            const predictions = model.predict(X);
            const predValues = await predictions.array();

            expect(predictions.shape).toEqual([4, 1]);
            expect(predValues[0][0]).toBe(0);
            expect(predValues[1][0]).toBe(0);
            expect(predValues[2][0]).toBe(1);
            expect(predValues[3][0]).toBe(1);

            X.dispose();
            predictions.dispose();
        });

        it('should throw error when model is not trained', () => {
            const untrainedModel = new QuadraticNaiveBayes({ regularization: 1e-9 });
            const X = tf.tensor2d([[1, 2]]);

            expect(() => {
                untrainedModel.predict(X);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
            untrainedModel.dispose?.();
        });
    });

    describe('sklearn validation tests', () => {
        it('should match sklearn QDA for simple 2-class problem', async () => {
            const X = tf.tensor2d([
                [1, 2],
                [1.5, 1.8],
                [5, 8],
                [8, 8],
                [1, 0.6],
                [9, 11],
            ]);
            const y = tf.tensor2d([[0], [0], [1], [1], [0], [1]]);

            const params = await model.train(X, y);

            // Test predictions on training data
            const testX = tf.tensor2d([
                [1, 2],
                [9, 11],
                [2, 2],
            ]);
            const predictions = model.predict(testX, params);
            const predValues = await predictions.array();

            // These values will be validated by the Python script
            expect(predValues[0][0]).toBe(0);
            expect(predValues[1][0]).toBe(1);

            X.dispose();
            y.dispose();
            testX.dispose();
            predictions.dispose();
        });

        it('should match sklearn QDA for 3-class iris subset', async () => {
            // Iris dataset subset (petal length, petal width)
            const X = tf.tensor2d([
                [1.4, 0.2],
                [1.4, 0.2],
                [1.3, 0.2],
                [4.7, 1.4],
                [4.5, 1.5],
                [4.9, 1.5],
                [6, 2.5],
                [5.1, 1.9],
                [5.9, 2.1],
            ]);
            const y = tf.tensor2d([[0], [0], [0], [1], [1], [1], [2], [2], [2]]);

            const params = await model.train(X, y);

            // Validate parameter shapes
            expect(params.classes.length).toBe(3);
            expect(params.classMeans.length).toBe(3);
            expect(params.classCovariances.length).toBe(3);

            // Test prediction
            const testX = tf.tensor2d([
                [1.4, 0.2],
                [4.7, 1.4],
                [6, 2.5],
            ]);
            const predictions = model.predict(testX, params);
            const predValues = await predictions.array();

            expect(predValues[0][0]).toBe(0);
            expect(predValues[1][0]).toBe(1);
            expect(predValues[2][0]).toBe(2);

            X.dispose();
            y.dispose();
            testX.dispose();
            predictions.dispose();
        });

        it('should handle quadratic decision boundaries', async () => {
            // Dataset with quadratic decision boundary
            // Class 0: close to origin
            // Class 1: far from origin (ring around class 0)
            const X = tf.tensor2d([
                [0, 0],
                [0.5, 0],
                [0, 0.5],
                [-0.5, 0],
                [0, -0.5], // Class 0: near origin
                [3, 0],
                [0, 3],
                [-3, 0],
                [0, -3],
                [2, 2],
                [-2, -2], // Class 1: far from origin
            ]);
            const y = tf.tensor2d([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]]);

            await model.train(X, y);

            // Test points
            const testX = tf.tensor2d([
                [0, 0],
                [3, 3],
                [0.2, 0.2],
            ]);
            const predictions = model.predict(testX);
            const predValues = await predictions.array();

            // Origin should be class 0
            expect(predValues[0][0]).toBe(0);
            // Far point should be class 1
            expect(predValues[1][0]).toBe(1);
            // Near origin should be class 0
            expect(predValues[2][0]).toBe(0);

            X.dispose();
            y.dispose();
            testX.dispose();
            predictions.dispose();
        });
    });
});
