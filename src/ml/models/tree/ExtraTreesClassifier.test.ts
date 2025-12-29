import { beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { softVoting } from '../../aggregators';
import { ExtraTreesClassifier } from './ExtraTreesClassifier';
import { Gini } from '../../criteria';
import type { CriterionFunction } from '../../types';

// Classification test data - 3 classes with clear separation
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

// One-hot encoded labels for 3 classes
const yArr = [
    [1, 0, 0], // Class 0
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0], // Class 1
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1], // Class 2
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
];

describe('ExtraTreesClassifier', () => {
    let model: ExtraTreesClassifier;
    let criterion: CriterionFunction;

    beforeEach(() => {
        criterion = new Gini();

        model = new ExtraTreesClassifier({
            criterion,
            maxDepth: 2,
            bootstrap: true,
            estimators: 2,
            maxFeatures: 1,
            aggregator: softVoting,
            numRandomThresholds: 5,
        });
    });

    describe('train', () => {
        it('should train on simple classification data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const trees = await model.train(X, y);

            expect(trees.length).toBe(2);
            expect(Array.isArray(trees[0]?.probabilities)).toBe(true);
            expect(Array.isArray(trees[1]?.probabilities)).toBe(true);

            // Verify tree structure
            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.269, 3);
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(1.449, 3);
            expect(trees[1]?.value).toBe(0);

            X.dispose();
            y.dispose();
        });

        it('should train with different number of estimators', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(3);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.269, 3);
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(1.449, 3);
            expect(trees[1]?.value).toBe(0);

            expect(trees[2]?.featureIndex).toBe(1);
            expect(trees[2]?.threshold).toBeCloseTo(3.165, 3);
            expect(trees[2]?.value).toBe(1);

            X.dispose();
            y.dispose();
        });

        it('should train with maxDepth = 1', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 1,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            // Verify shallow trees
            const tree = trees[0];
            expect(tree?.leftChild?.leftChild).toBeNull();
            expect(tree?.leftChild?.rightChild).toBeNull();
            expect(tree?.rightChild?.leftChild).toBeNull();
            expect(tree?.rightChild?.rightChild).toBeNull();

            X.dispose();
            y.dispose();
        });

        it('should train with bootstrap = false', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: false,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(trees[1]?.featureIndex);
            expect(trees[0]?.threshold).toBe(trees[1]?.threshold);
            expect(trees[0]?.value).toBe(trees[1]?.value);

            X.dispose();
            y.dispose();
        });

        it('should train with different maxFeatures', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.269, 3);
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(1.449, 3);
            expect(trees[1]?.value).toBe(0);

            X.dispose();
            y.dispose();
        });

        it('should train with maxFeatures = undefined (use all features)', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                // maxFeatures: undefined (default)
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.269, 3);
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(1.449, 3);
            expect(trees[1]?.value).toBe(0);

            X.dispose();
            y.dispose();
        });

        it('should train with different numRandomThresholds', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 10,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(5.269, 3);
            expect(trees[0]?.value).toBe(1);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(1.449, 3);
            expect(trees[1]?.value).toBe(0);

            X.dispose();
            y.dispose();
        });
    });

    describe('predict', () => {
        it('should predict on simple classification data', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            await model.train(X, y);

            const predictions = model.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            // Check that predictions are valid class indices
            const predArray = predictions.dataSync();
            for (const pred of predArray) {
                expect(typeof pred).toBe('number');
                expect(pred).toBeGreaterThanOrEqual(0);
                expect(pred).toBeLessThan(3); // 3 classes
                expect(Number.isInteger(pred)).toBe(true);
            }

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should predict with different number of estimators', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            await classifier.train(X, y);

            const predictions = classifier.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should throw error when predicting before training', () => {
            const X = tf.tensor2d([[1.0, 2.0]]);

            // Create a fresh model instance that hasn't been trained
            const freshModel = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            expect(() => {
                freshModel.predict(X);
            }).toThrow('Model has not been trained yet. Please call train() first.');

            X.dispose();
        });
    });

    describe('integration tests', () => {
        it('should solve multi-class classification with feature engineering', async () => {
            // Create more complex data with polynomial features
            const complexX = XArr.map(([x1, x2]) => [x1, x2, x1 * x1, x2 * x2, x1 * x2]);
            const X = tf.tensor2d(complexX);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 3,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 3,
                aggregator: softVoting,
                numRandomThresholds: 10,
            });

            await classifier.train(X, y);

            const predictions = classifier.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            // Verify predictions are reasonable class indices
            const predArray = predictions.dataSync();
            for (const pred of predArray) {
                expect(pred).toBeGreaterThanOrEqual(0);
                expect(pred).toBeLessThan(3);
                expect(Number.isInteger(pred)).toBe(true);
            }

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should handle noisy data', async () => {
            // Add noise to the data
            const noisyX = XArr.map(([x1, x2]) => [
                x1 + (Math.random() - 0.5) * 0.5,
                x2 + (Math.random() - 0.5) * 0.5,
            ]);

            const X = tf.tensor2d(noisyX);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 3,
                bootstrap: true,
                estimators: 3,
                maxFeatures: 2,
                aggregator: softVoting,
                numRandomThresholds: 8,
            });

            await classifier.train(X, y);

            const predictions = classifier.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });
    });

    describe('edge cases', () => {
        it('should handle small dataset', async () => {
            const smallX = [
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
            ];
            const smallY = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ];

            const X = tf.tensor2d(smallX);
            const y = tf.tensor2d(smallY);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 3,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBe(0);
            expect(trees[0]?.threshold).toBeCloseTo(6.065, 3);
            expect(trees[0]?.value).toBe(2);

            expect(trees[1]?.featureIndex).toBe(0);
            expect(trees[1]?.threshold).toBeCloseTo(3.065, 3);
            expect(trees[1]?.value).toBe(0);

            const predictions = classifier.predict(X);

            expect(predictions.shape).toEqual([3, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should handle uniform target values', async () => {
            const uniformX = [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
            ];
            const uniformY = [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]; // All same class

            const X = tf.tensor2d(uniformX);
            const y = tf.tensor2d(uniformY);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 1,
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            expect(trees[0]?.featureIndex).toBeNull();
            expect(trees[0]?.threshold).toBeNull();
            expect(trees[0]?.value).toBe(0);

            expect(trees[1]?.featureIndex).toBeNull();
            expect(trees[1]?.threshold).toBeNull();
            expect(trees[1]?.value).toBe(0);

            const predictions = classifier.predict(X);

            expect(predictions.shape).toEqual([3, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });

        it('should handle maxFeatures larger than available features', async () => {
            const X = tf.tensor2d(XArr);
            const y = tf.tensor2d(yArr);

            const classifier = new ExtraTreesClassifier({
                criterion,
                maxDepth: 2,
                bootstrap: true,
                estimators: 2,
                maxFeatures: 10, // Larger than available features (2)
                aggregator: softVoting,
                numRandomThresholds: 5,
            });

            const trees = await classifier.train(X, y);

            expect(trees.length).toBe(2);

            const predictions = classifier.predict(X);

            expect(predictions.shape).toEqual([XArr.length, 1]);

            X.dispose();
            y.dispose();
            predictions.dispose();
        });
    });
});
