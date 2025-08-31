import { beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { CriterionFunction } from '../../types';
import { RandomForestRegressor } from './RandomForestRegressor';
import { MeanSquaredError } from '../../criteria';
import { avgPreds } from '../../aggregators';

const XArr = [
    0.0005719, 0.0914414, 0.0968348, 0.136938, 0.1952739, 0.2497673, 0.2668127, 0.4252211, 0.461693,
    0.4917342, 0.5116721, 0.51613, 0.6501429, 0.6873735, 0.6963817, 0.7019347, 0.7336429, 0.7337795,
    0.826771, 0.8491521, 0.9313011, 0.9905074, 1.0222612, 1.0581406, 1.3277333, 1.40222, 1.4388767,
    1.4680707, 1.5116629, 1.5671209, 1.5775782, 1.7278036, 1.7388293, 1.9838374, 1.9883842,
    2.0702799, 2.0708963, 2.08511, 2.086524, 2.0959726, 2.1055381, 2.2394676, 2.4578658, 2.5744456,
    2.6658264, 2.679482, 2.6940837, 2.7934491, 2.870588, 2.9327752, 2.9465277, 3.3189732, 3.3523376,
    3.3941777, 3.4260975, 3.4325046, 3.4593856, 3.4616131, 3.4720008, 3.4987918, 3.6016225,
    3.6299899, 3.7408283, 3.7507216, 3.7540605, 3.9463966, 4.0037228, 4.0369564, 4.1731284,
    4.3819458, 4.3905872, 4.3907125, 4.4730333, 4.5170096, 4.5429775, 4.6375429, 4.7229738,
    4.7894477, 4.8413079, 4.9443054,
].map((v) => [v]);

const yArr = [
    -0.3827342, 0.091314, 0.0966835, 0.1365104, 0.1940353, 0.1235063, 0.2636583, 0.4125222,
    0.4454645, 0.4721553, 0.2386935, 0.493518, 0.6053001, 0.6345093, 0.6414461, 0.7967979,
    0.6695798, 0.6696812, 0.7357483, 0.7507205, 1.0324692, 0.8363043, 0.8532893, 0.871445,
    0.9706053, 0.5899384, 0.9913112, 0.9947284, 0.9982521, 0.9999932, 1.0718858, 0.9876997,
    0.9859156, 0.9159044, 0.9140699, 0.41299, 0.8775346, 0.8706305, 0.8699341, 0.8652356, 0.6969587,
    0.7846461, 0.6316866, 0.537228, 0.4580197, 0.3241426, 0.4327212, 0.3411533, 0.2676995,
    0.2073032, 0.5790843, -0.1764519, -0.2091884, -0.2499078, -0.2806822, -0.7363153, -0.3124708,
    -0.3145859, -0.3244291, -0.3496515, -0.393887, -0.4692111, -0.5640114, -0.5721533, -0.5748885,
    -0.7990844, -0.7592307, -0.7804366, -0.8580886, -0.9458986, -0.8568039, -0.9487067, -0.9714909,
    -0.9809741, -0.9856842, -0.7342273, -0.999944, -0.9970324, -0.9917015, -0.9732277,
].map((v) => [v]);

describe('RandomForestRegressor', () => {
    let model: RandomForestRegressor;

    let criterion: CriterionFunction;

    beforeEach(() => {
        criterion = new MeanSquaredError();

        model = new RandomForestRegressor({
            criterion,
            maxDepth: 2,
            bootstrap: true,
            estimators: 2,
            maxFeatures: 1,
            aggregator: avgPreds,
        });
    });

    describe('train', () => {
        it('should train on simple regression data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await model.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(3.133, 3);
            expect(trees[0]?.value).toBeCloseTo(0.101, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(3.126, 3);
            expect(trees[1]?.value).toBeCloseTo(0.066, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with different number of estimators', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            const trees = await regressor.train(X, y);

            expect(trees.length).toBe(3);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(3.133, 3);
            expect(trees[0]?.value).toBeCloseTo(0.101, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(3.126, 3);
            expect(trees[1]?.value).toBeCloseTo(0.066, 3);

            expect(trees[2]?.featureIndex).toBe(0);
            expect(trees[2]?.threshold).toBeCloseTo(3.095, 3);
            expect(trees[2]?.value).toBeCloseTo(0.297, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 1', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 1,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            const trees = await regressor.train(X, y);

            expect(trees.length).toBe(2);

            // Check that trees have depth 1 (no grandchildren)
            expect(trees[0]?.leftChild?.leftChild).toBeNull();
            expect(trees[0]?.leftChild?.rightChild).toBeNull();
            expect(trees[0]?.rightChild?.leftChild).toBeNull();
            expect(trees[0]?.rightChild?.rightChild).toBeNull();

            X.dispose();
            y.dispose();
        });

        it('should train with bootstrap = false', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: false,
                estimators: 2,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            const trees = await regressor.train(X, y);

            expect(trees[0]?.featureIndex).toBe(trees[1]?.featureIndex);
            expect(trees[0]?.threshold).toBe(trees[1]?.threshold);
            expect(trees[0]?.value).toBe(trees[1]?.value);

            expect(trees.length).toBe(2);

            X.dispose();
            y.dispose();
        });
    });

    describe('predict', () => {
        it('should predict on simple regression data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const predictions = model.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            // Check that predictions are reasonable numbers
            const predArray = predictions.arraySync() as number[][];
            for (const pred of predArray) {
                expect(typeof pred[0]).toBe('number');
                expect(isNaN(pred[0])).toBe(false);
                expect(isFinite(pred[0])).toBe(true);
            }

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should predict with different number of estimators', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            await regressor.train(X, y);

            const predictions = regressor.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should predict on training data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const predictions = model.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should predict with single sample', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const singleSample = tf.tensor2d([[2.0]]);
            const predictions = model.predict(singleSample);

            expect(predictions.shape).toEqual([1, 1]);

            X.dispose();
            y.dispose();
            singleSample.dispose();
            predictions.dispose();
        });

        it('should throw error when predicting before training', () => {
            const X = tf.tensor2d([[1.0]]);

            // Create a completely fresh model instance for this test
            const freshModel = new RandomForestRegressor({
                criterion: new MeanSquaredError(),
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            expect(() => {
                freshModel.predict(X);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
        });
    });

    describe('integration tests', () => {
        it('should solve polynomial regression with feature engineering', async () => {
            // Create polynomial data: y = x^2 + noise
            const polyX = [];
            const polyY = [];

            for (let i = 0; i < 50; i++) {
                const x = (i - 25) / 25; // Range from -1 to 1
                const y = x * x + (Math.random() - 0.5) * 0.1; // x^2 with noise
                polyX.push([x]);
                polyY.push([y]);
            }

            const X = tf.tensor2d(polyX);
            const y = tf.tensor2d(polyY);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 4,
                bootstrap: true,
                estimators: 5,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            await regressor.train(X, y);

            const predictions = regressor.predict(X);

            expect(predictions.shape).toEqual([50, 1]);

            // Check that predictions are reasonable
            const predArray = predictions.arraySync() as number[][];
            for (const pred of predArray) {
                expect(typeof pred[0]).toBe('number');
                expect(isNaN(pred[0])).toBe(false);
                expect(isFinite(pred[0])).toBe(true);
            }

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should handle noisy data', async () => {
            // Create data with significant noise
            const noisyX = [];
            const noisyY = [];

            for (let i = 0; i < 30; i++) {
                const x = i / 10;
                const y = Math.sin(x) + (Math.random() - 0.5) * 2; // Sine wave with lots of noise
                noisyX.push([x]);
                noisyY.push([y]);
            }

            const X = tf.tensor2d(noisyX);
            const y = tf.tensor2d(noisyY);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 3,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            await regressor.train(X, y);

            const predictions = regressor.predict(X);

            expect(predictions.shape).toEqual([30, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });
    });

    describe('edge cases', () => {
        it('should handle small dataset', async () => {
            const smallX = [[1.0], [2.0], [3.0]];
            const smallY = [[1.0], [2.0], [3.0]];

            const X = tf.tensor2d(smallX);
            const y = tf.tensor2d(smallY);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 1,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            const trees = await regressor.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBe(2.5);
            expect(trees[0]?.value).toBeCloseTo(2.667, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBe(1.5);
            expect(trees[1]?.value).toBeCloseTo(1.333, 3);

            const predictions = regressor.predict(X);

            expect(predictions.shape).toEqual([3, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should handle uniform target values', async () => {
            const uniformX = [[1.0], [2.0], [3.0], [4.0], [5.0]];
            const uniformY = [[1.0], [1.0], [1.0], [1.0], [1.0]];

            const X = tf.tensor2d(uniformX);
            const y = tf.tensor2d(uniformY);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: avgPreds,
            });

            const trees = await regressor.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBeNull();
            expect(trees[0]?.threshold).toBeNull();
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBeNull();
            expect(trees[1]?.threshold).toBeNull();
            expect(trees[1]?.value).toBe(1);

            const predictions = regressor.predict(X);

            expect(predictions.shape).toEqual([5, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should handle maxFeatures larger than available features', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const regressor = new RandomForestRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 10, // Larger than available features (1)
                aggregator: avgPreds,
            });

            const trees = await regressor.train(X, y);

            expect(trees.length).toBe(2);

            const predictions = regressor.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });
    });
});
