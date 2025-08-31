import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { CriterionFunction } from '../../types';
import { ExtraTreesRegressor } from './ExtraTreesRegressor';
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

describe('ExtraTreesRegressor', () => {
    let model: ExtraTreesRegressor;

    let criterion: CriterionFunction;

    beforeEach(() => {
        criterion = new MeanSquaredError();

        model = new ExtraTreesRegressor({
            criterion,
            maxDepth: 2,
            bootstrap: true,
            estimators: 2,
            aggregator: avgPreds,
            numRandomThresholds: 5,
        });
    });

    describe('train', () => {
        it('should train on simple linear data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await model.train(X, y);

            expect(trees.length).toBe(2);

            let tree = trees[0];

            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(3.355, 3);
            expect(tree?.value).toBeCloseTo(0.101, 3);

            expect(tree?.leftChild?.featureIndex).toBe(0);
            expect(tree?.leftChild?.threshold).toBeCloseTo(2.439, 3);
            expect(tree?.leftChild?.value).toBeCloseTo(0.636, 3);

            expect(tree?.rightChild?.featureIndex).toBe(0);
            expect(tree?.rightChild?.threshold).toBeCloseTo(3.898, 3);
            expect(tree?.rightChild?.value).toBeCloseTo(-0.66, 3);

            expect(tree?.rightChild?.leftChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.leftChild?.threshold).toBe(null);
            expect(tree?.rightChild?.leftChild?.value).toBeCloseTo(-0.331, 3);

            expect(tree?.rightChild?.rightChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.rightChild?.threshold).toBe(null);
            expect(tree?.rightChild?.rightChild?.value).toBeCloseTo(-0.873, 3);

            tree = trees[1];

            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(3.29, 3);
            expect(tree?.value).toBeCloseTo(0.066, 3);

            expect(tree?.leftChild?.featureIndex).toBe(0);
            expect(tree?.leftChild?.threshold).toBeCloseTo(0.457, 3);
            expect(tree?.leftChild?.value).toBeCloseTo(0.593, 3);

            expect(tree?.rightChild?.featureIndex).toBe(0);
            expect(tree?.rightChild?.threshold).toBeCloseTo(4.035, 3);
            expect(tree?.rightChild?.value).toBeCloseTo(-0.612, 3);

            expect(tree?.rightChild?.leftChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.leftChild?.threshold).toBe(null);
            expect(tree?.rightChild?.leftChild?.value).toBeCloseTo(-0.338, 3);

            expect(tree?.rightChild?.rightChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.rightChild?.threshold).toBe(null);
            expect(tree?.rightChild?.rightChild?.value).toBeCloseTo(-0.938, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with different number of estimators', async () => {
            const modelFew = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 1,
                aggregator: avgPreds,
                numRandomThresholds: 5,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelFew.train(X, y);

            expect(trees.length).toBe(1);
            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(3.355, 3);
            expect(trees[0]?.value).toBeCloseTo(0.101, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with more estimators', async () => {
            const modelMany = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 5,
                aggregator: avgPreds,
                numRandomThresholds: 5,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelMany.train(X, y);

            expect(trees.length).toBe(5);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(3.355, 3);
            expect(trees[0]?.value).toBeCloseTo(0.101, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(3.29, 3);
            expect(trees[1]?.value).toBeCloseTo(0.066, 3);

            expect(trees[2]?.featureIndex).toBe(0);
            expect(trees[2]?.threshold).toBeCloseTo(3.084, 3);
            expect(trees[2]?.value).toBeCloseTo(0.297, 3);

            expect(trees[3]?.featureIndex).toBe(0);
            expect(trees[3]?.threshold).toBeCloseTo(2.825, 3);
            expect(trees[3]?.value).toBeCloseTo(0.19, 3);

            expect(trees[4]?.featureIndex).toBe(0);
            expect(trees[4]?.threshold).toBeCloseTo(3.03, 3);
            expect(trees[4]?.value).toBeCloseTo(0.066, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 1', async () => {
            const modelShallow = new ExtraTreesRegressor({
                criterion,
                maxDepth: 1,
                bootstrap: true,
                estimators: 2,
                aggregator: avgPreds,
                numRandomThresholds: 5,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelShallow.train(X, y);

            expect(trees.length).toBe(2);
            trees.forEach((tree) => {
                expect(tree).toBeDefined();
                // With maxDepth=1, trees should have limited depth
                expect(tree?.leftChild?.featureIndex).toBe(null);
                expect(tree?.rightChild?.featureIndex).toBe(null);
            });

            X.dispose();
            y.dispose();
        });

        it('should train with bootstrap = false', async () => {
            const modelNoBootstrap = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: false,
                estimators: 2,
                aggregator: avgPreds,
                numRandomThresholds: 5,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelNoBootstrap.train(X, y);

            expect(trees.length).toBe(2);
            // Without bootstrap, trees should be identical
            expect(trees[0]?.featureIndex).toBe(trees[1]?.featureIndex);
            expect(trees[0]?.threshold).toBe(trees[1]?.threshold);
            expect(trees[0]?.value).toBe(trees[1]?.value);

            X.dispose();
            y.dispose();
        });

        it('should train with different numRandomThresholds', async () => {
            const modelFewThresholds = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: avgPreds,
                numRandomThresholds: 3,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelFewThresholds.train(X, y);

            expect(trees.length).toBe(2);
            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(2.157, 3);
            expect(trees[0]?.value).toBeCloseTo(0.101, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(2.766, 3);
            expect(trees[1]?.value).toBeCloseTo(0.066, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with more numRandomThresholds', async () => {
            const modelManyThresholds = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: avgPreds,
                numRandomThresholds: 10,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelManyThresholds.train(X, y);

            expect(trees.length).toBe(2);
            trees.forEach((tree) => {
                expect(tree).toBeDefined();
            });

            X.dispose();
            y.dispose();
        });

        it('should handle small dataset', async () => {
            const XSmall = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
            const ySmall = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);

            const modelSmall = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: avgPreds,
                numRandomThresholds: 5,
            });

            const trees = await modelSmall.train(XSmall, ySmall);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(2.211, 3);
            expect(trees[0]?.value).toBeCloseTo(2.75, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(2.114, 3);
            expect(trees[1]?.value).toBeCloseTo(2.5, 3);

            XSmall.dispose();
            ySmall.dispose();
        });

        it('should train on data with multiple features', async () => {
            const XMulti = tf.tensor2d([
                [1.0, 2.0],
                [1.1, 2.1],
                [4.0, 5.0],
                [4.1, 5.1],
                [7.0, 8.0],
                [7.1, 8.1],
            ]);
            const yMulti = tf.tensor2d([[1.0], [1.1], [4.0], [4.1], [7.0], [7.1]]);

            const modelMulti = new ExtraTreesRegressor({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: avgPreds,
                numRandomThresholds: 5,
            });

            const trees = await modelMulti.train(XMulti, yMulti);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(6.934, 3);
            expect(trees[0]?.value).toBeCloseTo(5.083, 3);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(1.94, 3);
            expect(trees[1]?.value).toBeCloseTo(3.583, 3);

            XMulti.dispose();
            yMulti.dispose();
        });

        describe('predict', () => {
            it('should predict on simple linear data', async () => {
                // y = cos(x)
                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const tree = await model.train(X, y);
                expect(tree).toBeDefined();

                const X_test = tf.tensor2d([
                    [7.4, 8.0],
                    [1.0, 2.0],
                    [4.2, 4.9],
                ]);

                const y_pred = model.predict(X_test);

                expect(y_pred.dataSync()[0]).toBeCloseTo(-0.905, 3);
                expect(y_pred.dataSync()[1]).toBeCloseTo(0.727, 3);
                expect(y_pred.dataSync()[2]).toBeCloseTo(-0.905, 3);

                X.dispose();
                y.dispose();
            });

            it('should predict with different number of estimators', async () => {
                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const modelFew = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: true,
                    estimators: 1,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                await modelFew.train(X, y);

                const XTest = tf.tensor2d([[0.5], [2.0]]);

                const yPred = modelFew.predict(XTest);

                expect(yPred).toBeDefined();
                expect(yPred.shape[0]).toBe(2);

                X.dispose();
                y.dispose();
                XTest.dispose();
            });

            it('should predict on training data', async () => {
                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                await model.train(X, y);

                const yPred = model.predict(X);

                expect(yPred).toBeDefined();
                expect(yPred.shape[0]).toBe(XArr.length);

                // Check that predictions are reasonable (not NaN or infinite)
                const predValues = yPred.dataSync();
                predValues.forEach((pred) => {
                    expect(isFinite(pred)).toBe(true);
                    expect(isNaN(pred)).toBe(false);
                });

                X.dispose();
                y.dispose();
            });

            it('should predict with single sample', async () => {
                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                await model.train(X, y);

                const XSingle = tf.tensor2d([[2.5]]);
                const yPred = model.predict(XSingle);

                expect(yPred).toBeDefined();
                expect(yPred.shape).toEqual([1, 1]);
                expect(isFinite(yPred.dataSync()[0])).toBe(true);

                X.dispose();
                y.dispose();
                XSingle.dispose();
            });

            it('should predict with bootstrap = false', async () => {
                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const modelNoBootstrap = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: false,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                await modelNoBootstrap.train(X, y);

                const XTest = tf.tensor2d([[0.5], [2.0], [4.0]]);

                const yPred = modelNoBootstrap.predict(XTest);

                expect(yPred).toBeDefined();
                expect(yPred.shape[0]).toBe(3);

                X.dispose();
                y.dispose();
                XTest.dispose();
            });

            it('should predict with different numRandomThresholds', async () => {
                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const modelFewThresholds = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: true,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 3,
                });

                await modelFewThresholds.train(X, y);

                const XTest = tf.tensor2d([[1.0], [3.0]]);

                const yPred = modelFewThresholds.predict(XTest);

                expect(yPred).toBeDefined();
                expect(yPred.shape[0]).toBe(2);

                X.dispose();
                y.dispose();
                XTest.dispose();
            });

            it('should handle prediction on small dataset', async () => {
                const XSmall = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
                const ySmall = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);

                const modelSmall = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: true,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                await modelSmall.train(XSmall, ySmall);

                const XTestSmall = tf.tensor2d([[2.5]]);
                const yPredSmall = modelSmall.predict(XTestSmall);

                expect(yPredSmall).toBeDefined();
                expect(yPredSmall.shape[0]).toBe(1);

                XSmall.dispose();
                ySmall.dispose();
                XTestSmall.dispose();
            });

            it('should predict on different data distributions', async () => {
                const XLinear = tf.tensor2d([[1.0], [2.0], [3.0], [4.0], [5.0]]);
                const yLinear = tf.tensor2d([[2.0], [4.0], [6.0], [8.0], [10.0]]);

                const modelLinear = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 3,
                    bootstrap: true,
                    estimators: 3,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                await modelLinear.train(XLinear, yLinear);

                const XTestLinear = tf.tensor2d([[1.5], [3.5]]);
                const yPredLinear = modelLinear.predict(XTestLinear);

                expect(yPredLinear).toBeDefined();
                expect(yPredLinear.shape[0]).toBe(2);

                XLinear.dispose();
                yLinear.dispose();
                XTestLinear.dispose();
            });
        });

        describe('edge cases', () => {
            it('should handle single estimator', async () => {
                const modelSingle = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: true,
                    estimators: 1,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const trees = await modelSingle.train(X, y);
                expect(trees.length).toBe(1);

                const XTest = tf.tensor2d([[2.5]]);
                const yPred = modelSingle.predict(XTest);

                expect(yPred).toBeDefined();
                expect(yPred.shape[0]).toBe(1);

                X.dispose();
                y.dispose();
                XTest.dispose();
            });

            it('should handle maxDepth = 0', async () => {
                const modelRoot = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 0,
                    bootstrap: true,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const trees = await modelRoot.train(X, y);

                expect(trees.length).toBe(2);
                trees.forEach((tree) => {
                    expect(tree).toBeDefined();
                    expect(tree?.leftChild).toBe(null);
                    expect(tree?.rightChild).toBe(null);
                });

                X.dispose();
                y.dispose();
            });

            it('should handle uniform target values', async () => {
                const XUniform = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
                const yUniform = tf.tensor2d([[5.0], [5.0], [5.0], [5.0]]);

                const modelUniform = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: true,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                const trees = await modelUniform.train(XUniform, yUniform);

                expect(trees.length).toBe(2);
                trees.forEach((tree) => {
                    expect(tree).toBeDefined();
                });

                XUniform.dispose();
                yUniform.dispose();
            });

            it('should handle very small dataset', async () => {
                const XSmall = tf.tensor2d([[1.0], [2.0]]);
                const ySmall = tf.tensor2d([[10.0], [20.0]]);

                const modelSmall = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 1,
                    bootstrap: true,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 5,
                });

                const trees = await modelSmall.train(XSmall, ySmall);

                expect(trees.length).toBe(2);
                trees.forEach((tree) => {
                    expect(tree).toBeDefined();
                });

                XSmall.dispose();
                ySmall.dispose();
            });

            it('should handle numRandomThresholds = 1', async () => {
                const modelMinThresholds = new ExtraTreesRegressor({
                    criterion,
                    maxDepth: 2,
                    bootstrap: true,
                    estimators: 2,
                    aggregator: avgPreds,
                    numRandomThresholds: 1,
                });

                const X = tf.tensor2d(XArr);
                const y = tf.tensor2d(yArr);

                const trees = await modelMinThresholds.train(X, y);

                expect(trees.length).toBe(2);
                trees.forEach((tree) => {
                    expect(tree).toBeDefined();
                });

                X.dispose();
                y.dispose();
            });
        });

        afterEach(() => {
            model.dispose?.();
        });
    });
});
