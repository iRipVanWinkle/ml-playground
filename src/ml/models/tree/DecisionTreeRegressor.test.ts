import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { DecisionTreeRegressor } from './DecisionTreeRegressor';
import type { CriterionFunction } from '../../types';
import { MeanSquaredError } from '../../criteria';

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

describe('DecisionTreeRegressor', () => {
    let model: DecisionTreeRegressor;

    let criterion: CriterionFunction;

    beforeEach(() => {
        criterion = new MeanSquaredError();

        model = new DecisionTreeRegressor({ criterion });
    });

    describe('train', () => {
        it('should train on simple linear data', async () => {
            // y = cos(x)
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await model.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();

            expect(tree?.threshold).toBeCloseTo(3.133, 3);
            expect(tree?.value).toBeCloseTo(0.152, 3);

            expect(tree?.leftChild?.threshold).toBeCloseTo(0.514, 3);
            expect(tree?.leftChild?.value).toBeCloseTo(0.613, 3);

            expect(tree?.leftChild?.leftChild?.threshold).toBeCloseTo(0.046, 3);
            expect(tree?.leftChild?.leftChild?.value).toBeCloseTo(0.19, 3);

            expect(tree?.rightChild?.threshold).toBeCloseTo(3.85, 3);
            expect(tree?.rightChild?.value).toBeCloseTo(-0.659, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 1', async () => {
            const modelShallow = new DecisionTreeRegressor({ criterion, maxDepth: 1 });
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelShallow.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.threshold).toBeCloseTo(3.133, 3);
            expect(tree?.value).toBeCloseTo(0.152, 3);

            // With maxDepth=1, children should be leaf nodes
            expect(tree?.leftChild?.featureIndex).toBe(null);
            expect(tree?.leftChild?.threshold).toBe(null);
            expect(tree?.leftChild?.value).toBeCloseTo(0.613, 3);

            expect(tree?.rightChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.threshold).toBe(null);
            expect(tree?.rightChild?.value).toBeCloseTo(-0.659, 3);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 3', async () => {
            const modelMedium = new DecisionTreeRegressor({ criterion, maxDepth: 3 });
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelMedium.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.threshold).toBeCloseTo(3.133, 3);

            // Should have deeper structure than maxDepth=1
            expect(tree?.leftChild?.threshold).toBeCloseTo(0.514, 3);
            expect(tree?.rightChild?.threshold).toBeCloseTo(3.85, 3);

            X.dispose();
            y.dispose();
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

            const modelMulti = new DecisionTreeRegressor({ criterion, maxDepth: 2 });
            const trees = await modelMulti.train(XMulti, yMulti);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.featureIndex).toBeDefined();
            expect(tree?.threshold).toBeDefined();

            XMulti.dispose();
            yMulti.dispose();
        });

        it('should handle small dataset', async () => {
            const XSmall = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
            const ySmall = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);

            const modelSmall = new DecisionTreeRegressor({ criterion, maxDepth: 2 });
            const trees = await modelSmall.train(XSmall, ySmall);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.value).toBeCloseTo(2.5, 1); // Mean of 1,2,3,4

            XSmall.dispose();
            ySmall.dispose();
        });
    });

    describe('predict', () => {
        it('should predict on simple linear data', async () => {
            // y = cos(x)
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const model2 = new DecisionTreeRegressor({ criterion, maxDepth: 2 });
            const model5 = new DecisionTreeRegressor({ criterion, maxDepth: 5 });

            const tree2 = await model2.train(X, y);
            const tree5 = await model5.train(X, y);

            expect(tree2).toBeDefined();
            expect(tree5).toBeDefined();

            const X_test = tf.tensor2d([[0.06], [0.824], [2.528], [3.137]]);

            const y_pred2 = model2.predict(X_test);
            const y_pred5 = model5.predict(X_test);

            expect(y_pred2.dataSync()[0]).toBeCloseTo(0.19, 3);
            expect(y_pred2.dataSync()[1]).toBeCloseTo(0.729, 3);
            expect(y_pred2.dataSync()[2]).toBeCloseTo(0.729, 3);
            expect(y_pred2.dataSync()[3]).toBeCloseTo(-0.395, 3);

            expect(y_pred5.dataSync()[0]).toBeCloseTo(0.094, 3);
            expect(y_pred5.dataSync()[1]).toBeCloseTo(0.725, 3);
            expect(y_pred5.dataSync()[2]).toBeCloseTo(0.498, 3);
            expect(y_pred5.dataSync()[3]).toBeCloseTo(-0.176, 3);

            X.dispose();
            y.dispose();
        });

        it('should predict on training data with high accuracy', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const modelDeep = new DecisionTreeRegressor({ criterion, maxDepth: 10 });
            await modelDeep.train(X, y);

            const yPred = modelDeep.predict(X);

            expect(yPred).toBeDefined();
            expect(yPred.shape[0]).toBe(XArr.length);

            // With deep tree, should achieve very low error on training data
            const mse = tf.metrics.meanSquaredError(y.flatten(), yPred.flatten());
            expect(mse.dataSync()[0]).toBeLessThan(0.01); // Very low MSE on training data

            X.dispose();
            y.dispose();
            yPred.dispose();
            mse.dispose();
        });

        it('should predict with maxDepth = 1', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const modelShallow = new DecisionTreeRegressor({ criterion, maxDepth: 1 });
            await modelShallow.train(X, y);

            const XTest = tf.tensor2d([[0.5], [2.0], [4.0]]);
            const yPred = modelShallow.predict(XTest);

            expect(yPred).toBeDefined();
            expect(yPred.shape[0]).toBe(3);

            // All predictions should be either left or right child values
            const predValues = yPred.dataSync();
            for (const pred of predValues) {
                const isLeftValue = Math.abs(pred - 0.613) < 0.1;
                const isRightValue = Math.abs(pred + 0.659) < 0.1;
                expect(isLeftValue || isRightValue).toBe(true);
            }

            X.dispose();
            y.dispose();
            XTest.dispose();
            yPred.dispose();
        });

        it('should handle prediction on single sample', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const XSingle = tf.tensor2d([[2.5]]);
            const yPred = model.predict(XSingle);

            expect(yPred).toBeDefined();
            expect(yPred.shape).toEqual([1, 1]);

            X.dispose();
            y.dispose();
            XSingle.dispose();
            yPred.dispose();
        });

        it('should predict on different data distributions', async () => {
            const XLinear = tf.tensor2d([[1.0], [2.0], [3.0], [4.0], [5.0]]);
            const yLinear = tf.tensor2d([[2.0], [4.0], [6.0], [8.0], [10.0]]);

            const modelLinear = new DecisionTreeRegressor({ criterion, maxDepth: 3 });
            await modelLinear.train(XLinear, yLinear);

            const XTestLinear = tf.tensor2d([[1.5], [3.5]]);
            const yPredLinear = modelLinear.predict(XTestLinear);

            expect(yPredLinear).toBeDefined();
            expect(yPredLinear.shape[0]).toBe(2);

            XLinear.dispose();
            yLinear.dispose();
            XTestLinear.dispose();
            yPredLinear.dispose();
        });
    });

    describe('edge cases', () => {
        it('should handle maxDepth = 0', async () => {
            const modelRoot = new DecisionTreeRegressor({ criterion, maxDepth: 0 });
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelRoot.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();
            // With maxDepth = 0, the tree shouldn't split, so no children
            expect(tree?.leftChild).toBe(null);
            expect(tree?.rightChild).toBe(null);
            expect(tree?.value).toBeDefined(); // Should have a prediction value
            expect(typeof tree?.value).toBe('number');

            // featureIndex and threshold might still be set even if no split occurs
            // depending on implementation
            expect(tree?.featureIndex).toBeDefined();
            expect(tree?.threshold).toBeDefined();

            X.dispose();
            y.dispose();
        });

        it('should handle uniform target values', async () => {
            const XUniform = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
            const yUniform = tf.tensor2d([[5.0], [5.0], [5.0], [5.0]]);

            const modelUniform = new DecisionTreeRegressor({ criterion, maxDepth: 2 });
            const trees = await modelUniform.train(XUniform, yUniform);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.value).toBe(5.0); // All values are the same

            XUniform.dispose();
            yUniform.dispose();
        });

        it('should handle very small dataset', async () => {
            const XSmall = tf.tensor2d([[1.0], [2.0]]);
            const ySmall = tf.tensor2d([[10.0], [20.0]]);

            const modelSmall = new DecisionTreeRegressor({ criterion, maxDepth: 1 });
            const trees = await modelSmall.train(XSmall, ySmall);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.value).toBe(15.0); // Mean of 10 and 20

            XSmall.dispose();
            ySmall.dispose();
        });

        it('should handle single feature prediction', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const XTest = tf.tensor2d([[XArr[0][0]]]); // Single feature from first sample
            const yPred = model.predict(XTest);

            expect(yPred).toBeDefined();
            expect(yPred.shape).toEqual([1, 1]);
            expect(typeof yPred.dataSync()[0]).toBe('number');

            X.dispose();
            y.dispose();
            XTest.dispose();
            yPred.dispose();
        });
    });

    afterEach(() => {
        model.dispose?.();
    });
});
