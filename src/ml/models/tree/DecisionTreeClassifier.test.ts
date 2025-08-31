import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { CriterionFunction } from '../../types';
import { DecisionTreeClassifier } from './DecisionTreeClassifier';
import { Gini, Entropy } from '../../criteria';

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

describe('DecisionTreeClassifier', () => {
    let model: DecisionTreeClassifier;

    let criterion: CriterionFunction;

    beforeEach(() => {
        criterion = new Gini();

        model = new DecisionTreeClassifier({ criterion, maxDepth: 2 });
    });

    describe('train', () => {
        it('should train on simple linear data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await model.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();

            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(2.7, 2);
            expect(tree?.value).toBe(0);

            expect(tree?.leftChild?.featureIndex).toBe(null);
            expect(tree?.leftChild?.threshold).toBe(null);
            expect(tree?.leftChild?.value).toBe(0);

            expect(tree?.rightChild?.featureIndex).toBe(0);
            expect(tree?.rightChild?.threshold).toBeCloseTo(5.7, 2);
            expect(tree?.rightChild?.value).toBe(1);

            expect(tree?.rightChild?.leftChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.leftChild?.threshold).toBe(null);
            expect(tree?.rightChild?.leftChild?.value).toBe(1);

            expect(tree?.rightChild?.rightChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.rightChild?.threshold).toBe(null);
            expect(tree?.rightChild?.rightChild?.value).toBe(2);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 1', async () => {
            const modelShallow = new DecisionTreeClassifier({ criterion, maxDepth: 1 });
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelShallow.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(2.7, 2);
            expect(tree?.value).toBe(0);

            // With maxDepth=1, children should be leaf nodes (no further splits)
            expect(tree?.leftChild?.featureIndex).toBe(null);
            expect(tree?.leftChild?.threshold).toBe(null);
            expect(tree?.leftChild?.value).toBe(0);

            expect(tree?.rightChild?.featureIndex).toBe(null);
            expect(tree?.rightChild?.threshold).toBe(null);
            expect(tree?.rightChild?.value).toBe(1);

            // With maxDepth=1, there should be no grandchildren
            expect(tree?.leftChild?.leftChild).toBe(null);
            expect(tree?.leftChild?.rightChild).toBe(null);
            expect(tree?.rightChild?.leftChild).toBe(null);
            expect(tree?.rightChild?.rightChild).toBe(null);

            X.dispose();
            y.dispose();
        });

        it('should train on binary classification data', async () => {
            const XBinary = tf.tensor2d([
                [1.0, 2.0],
                [1.1, 2.0],
                [4.0, 5.0],
                [4.1, 5.1],
                [7.0, 8.0],
                [7.1, 8.0],
            ]);
            const yBinary = tf.tensor2d([
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ]);

            const modelBinary = new DecisionTreeClassifier({ criterion, maxDepth: 2 });
            const trees = await modelBinary.train(XBinary, yBinary);
            const tree = trees[0];

            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(2.55, 2);
            expect(tree?.value).toBe(1);

            XBinary.dispose();
            yBinary.dispose();
        });

        it('should train on data with three features', async () => {
            const XThree = tf.tensor2d([
                [1.0, 2.0, 3.0],
                [1.1, 2.0, 3.1],
                [4.0, 5.0, 6.0],
                [4.1, 5.1, 6.1],
                [7.0, 8.0, 9.0],
                [7.1, 8.0, 9.1],
            ]);
            const yThree = tf.tensor2d([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]);

            const modelThree = new DecisionTreeClassifier({ criterion, maxDepth: 2 });
            const trees = await modelThree.train(XThree, yThree);
            const tree = trees[0];

            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(2.55, 2);
            expect(tree?.value).toBe(0);

            XThree.dispose();
            yThree.dispose();
        });

        it('should train using Entropy criterion', async () => {
            const entropyCriterion = new Entropy();
            const modelEntropy = new DecisionTreeClassifier({
                criterion: entropyCriterion,
                maxDepth: 2,
            });
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await modelEntropy.train(X, y);
            const tree = trees[0];

            expect(tree).toBeDefined();
            expect(tree?.featureIndex).toBe(0);
            expect(tree?.threshold).toBeCloseTo(2.7, 2);

            X.dispose();
            y.dispose();
        });
    });

    describe('predict', () => {
        it('should predict on simple linear data', async () => {
            // y = cos(x)
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const model = new DecisionTreeClassifier({ criterion, maxDepth: 5 });

            const tree5 = await model.train(X, y);

            expect(tree5).toBeDefined();

            const XTest = tf.tensor2d([
                [7.4, 8.0],
                [1.0, 2.0],
                [4.2, 4.9],
            ]);

            const yPred5 = model.predict(XTest);

            expect(yPred5.dataSync()[0]).toBe(2);
            expect(yPred5.dataSync()[1]).toBe(0);
            expect(yPred5.dataSync()[2]).toBe(1);

            X.dispose();
            y.dispose();
        });

        it('should predict on training data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const model = new DecisionTreeClassifier({ criterion, maxDepth: 3 });
            await model.train(X, y);

            const yPred = model.predict(X);

            // Check that predictions match the training labels (argmax for multiclass)
            const yPredData = yPred.dataSync();
            const yData = y.argMax(1).dataSync();

            for (let i = 0; i < yPredData.length; i++) {
                expect(yPredData[i]).toBe(yData[i]);
            }

            X.dispose();
            y.dispose();
        });

        it('should predict on binary classification data', async () => {
            const XBinary = tf.tensor2d([
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
            ]);
            const yBinary = tf.tensor2d([
                [1, 0],
                [0, 1],
                [0, 1],
            ]);

            const modelBinary = new DecisionTreeClassifier({ criterion, maxDepth: 2 });
            await modelBinary.train(XBinary, yBinary);

            const XTestBinary = tf.tensor2d([
                [1.1, 2.1],
                [4.1, 5.1],
            ]);

            const yPredBinary = modelBinary.predict(XTestBinary);

            expect(yPredBinary.dataSync()[0]).toBe(0); // Should predict class 0
            expect(yPredBinary.dataSync()[1]).toBe(1); // Should predict class 1

            XBinary.dispose();
            yBinary.dispose();
            XTestBinary.dispose();
        });
    });

    afterEach(() => {
        model.dispose?.();
    });
});
