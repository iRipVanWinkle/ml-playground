import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { CriterionFunction } from '../../types';
import { BaggingClassifier } from './BaggingClassifier';
import { Gini } from '../../criteria';
import { softVoting } from '../../aggregators';

const XArr = [
    [1.0, 2.0],
    [1.1, 2.0],
    [1.2, 2.1],
    [1.3, 2.1],
    [1.4, 2.0],
    [4.0, 5.0],
    [4.1, 5.1],
    [4.2, 4.9],
    [4.3, 5.0],
    [4.4, 5.0],
    [7.0, 8.0],
    [7.1, 8.0],
    [7.2, 7.9],
    [7.3, 8.1],
    [7.4, 8.0],
];

const yArr = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
];

describe('BaggingClassifier', () => {
    let model: BaggingClassifier;

    let criterion: CriterionFunction;

    beforeEach(() => {
        criterion = new Gini();

        model = new BaggingClassifier({
            criterion,
            maxDepth: 2,
            bootstrap: true,
            estimators: 2,
            aggregator: softVoting,
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
            expect(tree?.threshold).toBeCloseTo(5.7, 3);
            expect(tree?.value).toBe(1);

            expect(tree?.leftChild?.featureIndex).toBe(0);
            expect(tree?.leftChild?.threshold).toBeCloseTo(2.7, 3);
            expect(tree?.leftChild?.value).toBe(1);

            expect(tree?.leftChild?.leftChild?.featureIndex).toBeNull();
            expect(tree?.leftChild?.leftChild?.threshold).toBeNull();
            expect(tree?.leftChild?.leftChild?.value).toBe(0);

            expect(tree?.leftChild?.rightChild?.featureIndex).toBeNull();
            expect(tree?.leftChild?.rightChild?.threshold).toBeNull();
            expect(tree?.leftChild?.rightChild?.value).toBe(1);

            expect(tree?.rightChild?.featureIndex).toBeNull();
            expect(tree?.rightChild?.threshold).toBeNull();
            expect(tree?.rightChild?.value).toBe(2);

            tree = trees[1];

            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(2.7, 3);
            expect(tree?.value).toBe(0);

            expect(tree?.leftChild?.featureIndex).toBeNull();
            expect(tree?.leftChild?.threshold).toBeNull();
            expect(tree?.leftChild?.value).toBe(0);

            expect(tree?.rightChild?.featureIndex).toBe(0);
            expect(tree?.rightChild?.threshold).toBeCloseTo(5.8, 3);
            expect(tree?.rightChild?.value).toBe(1);

            expect(tree?.rightChild?.leftChild?.featureIndex).toBeNull();
            expect(tree?.rightChild?.leftChild?.threshold).toBeNull();
            expect(tree?.rightChild?.leftChild?.value).toBe(1);

            expect(tree?.rightChild?.rightChild?.featureIndex).toBeNull();
            expect(tree?.rightChild?.rightChild?.threshold).toBeNull();
            expect(tree?.rightChild?.rightChild?.value).toBe(2);

            X.dispose();
            y.dispose();
        });

        it('should train with different number of estimators', async () => {
            const modelFew = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 1,
                aggregator: softVoting,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelFew.train(X, y);

            expect(trees.length).toBe(1);
            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.7, 3);
            expect(trees[0]?.value).toBe(1);

            X.dispose();
            y.dispose();
        });

        it('should train with more estimators', async () => {
            const modelMany = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 5,
                aggregator: softVoting,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelMany.train(X, y);

            expect(trees.length).toBe(5);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.7, 3);
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(2.7, 3);
            expect(trees[1]?.value).toBe(0);

            expect(trees[2]?.featureIndex).toBe(0);
            expect(trees[2]?.threshold).toBeCloseTo(2.7, 3);
            expect(trees[2]?.value).toBe(1);

            expect(trees[3]?.featureIndex).toBe(0);
            expect(trees[3]?.threshold).toBeCloseTo(2.65, 3);
            expect(trees[3]?.value).toBe(0);

            expect(trees[4]?.featureIndex).toBe(0);
            expect(trees[4]?.threshold).toBeCloseTo(2.6, 3);
            expect(trees[4]?.value).toBe(0);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 1', async () => {
            const modelShallow = new BaggingClassifier({
                criterion,
                maxDepth: 1,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
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
            const modelNoBootstrap = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: false,
                estimators: 2,
                aggregator: softVoting,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelNoBootstrap.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(2.7, 3);
            expect(trees[0]?.value).toBe(0);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(2.7, 3);
            expect(trees[1]?.value).toBe(0);

            X.dispose();
            y.dispose();
        });

        it('should handle small dataset', async () => {
            const XSmall = tf.tensor2d([
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ]);
            const ySmall = tf.tensor2d([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]);

            const modelSmall = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
            });

            const trees = await modelSmall.train(XSmall, ySmall);

            expect(trees.length).toBe(2);
            trees.forEach((tree) => {
                expect(tree).toBeDefined();
            });

            XSmall.dispose();
            ySmall.dispose();
        });

        it('should train on data with single feature', async () => {
            const XSingle = tf.tensor2d([[1.0], [1.1], [4.0], [4.1], [7.0], [7.1]]);
            const ySingle = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const modelSingle = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
            });

            const trees = await modelSingle.train(XSingle, ySingle);

            expect(trees.length).toBe(2);
            trees.forEach((tree) => {
                expect(tree).toBeDefined();
                expect(tree?.featureIndex).toBeDefined();
            });

            XSingle.dispose();
            ySingle.dispose();
        });
    });

    describe('predict', () => {
        it('should predict on simple linear data', async () => {
            // y = cos(x)
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const tree = await model.train(X, y);
            expect(tree).toBeDefined();

            const XTest = tf.tensor2d([
                [7.4, 8.0],
                [1.0, 2.0],
                [4.2, 4.9],
            ]);

            const yPred = model.predict(XTest);

            expect(yPred.dataSync()[0]).toBe(2);
            expect(yPred.dataSync()[1]).toBe(0);
            expect(yPred.dataSync()[2]).toBe(1);

            X.dispose();
            y.dispose();
        });

        it('should predict with different number of estimators', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const modelFew = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 1,
                aggregator: softVoting,
            });

            await modelFew.train(X, y);

            const XTest = tf.tensor2d([
                [1.0, 2.0],
                [4.0, 5.0],
            ]);

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

            // Check that predictions are valid class indices
            const predValues = yPred.dataSync();
            predValues.forEach((pred) => {
                expect([0, 1, 2]).toContain(pred);
            });

            X.dispose();
            y.dispose();
        });

        it('should predict with single sample', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const XSingle = tf.tensor2d([[2.5, 3.5]]);
            const yPred = model.predict(XSingle);

            expect(yPred).toBeDefined();
            expect(yPred.shape).toEqual([1, 1]);
            expect([0, 1, 2]).toContain(yPred.dataSync()[0]);

            X.dispose();
            y.dispose();
            XSingle.dispose();
        });

        it('should predict with bootstrap = false', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const modelNoBootstrap = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: false,
                estimators: 2,
                aggregator: softVoting,
            });

            await modelNoBootstrap.train(X, y);

            const XTest = tf.tensor2d([
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
            ]);

            const yPred = modelNoBootstrap.predict(XTest);

            expect(yPred).toBeDefined();
            expect(yPred.shape[0]).toBe(3);

            X.dispose();
            y.dispose();
            XTest.dispose();
        });

        it('should handle prediction on small dataset', async () => {
            const XSmall = tf.tensor2d([
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ]);
            const ySmall = tf.tensor2d([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]);

            const modelSmall = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
            });

            await modelSmall.train(XSmall, ySmall);

            const XTestSmall = tf.tensor2d([[2.5, 3.5]]);
            const yPredSmall = modelSmall.predict(XTestSmall);

            expect(yPredSmall).toBeDefined();
            expect(yPredSmall.shape[0]).toBe(1);

            XSmall.dispose();
            ySmall.dispose();
            XTestSmall.dispose();
        });

        it('should predict on different data distributions', async () => {
            const XLinear = tf.tensor2d([
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
            ]);
            const yLinear = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]);

            const modelLinear = new BaggingClassifier({
                criterion,
                maxDepth: 3,
                bootstrap: true,
                estimators: 3,
                aggregator: softVoting,
            });

            await modelLinear.train(XLinear, yLinear);

            const XTestLinear = tf.tensor2d([
                [1.5, 1.5],
                [3.5, 3.5],
            ]);
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
            const modelSingle = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 1,
                aggregator: softVoting,
            });

            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelSingle.train(X, y);
            expect(trees.length).toBe(1);

            const XTest = tf.tensor2d([[2.5, 3.5]]);
            const yPred = modelSingle.predict(XTest);

            expect(yPred).toBeDefined();
            expect(yPred.shape[0]).toBe(1);

            X.dispose();
            y.dispose();
            XTest.dispose();
        });

        it('should handle maxDepth = 0', async () => {
            const modelRoot = new BaggingClassifier({
                criterion,
                maxDepth: 0,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
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
            const XUniform = tf.tensor2d([
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ]);
            const yUniform = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]);

            const modelUniform = new BaggingClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
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
            const XSmall = tf.tensor2d([
                [1.0, 2.0],
                [2.0, 3.0],
            ]);
            const ySmall = tf.tensor2d([
                [1, 0, 0],
                [0, 1, 0],
            ]);

            const modelSmall = new BaggingClassifier({
                criterion,
                maxDepth: 1,
                bootstrap: true,
                estimators: 2,
                aggregator: softVoting,
            });

            const trees = await modelSmall.train(XSmall, ySmall);

            expect(trees.length).toBe(2);
            trees.forEach((tree) => {
                expect(tree).toBeDefined();
            });

            XSmall.dispose();
            ySmall.dispose();
        });
    });

    afterEach(() => {
        model.dispose?.();
    });
});
