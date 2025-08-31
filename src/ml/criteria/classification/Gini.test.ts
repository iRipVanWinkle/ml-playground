import { describe, it, expect, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { Gini } from './Gini';

describe('Gini', () => {
    let gini: Gini;

    beforeEach(() => {
        gini = new Gini();
    });

    describe('impurity', () => {
        it('should return 0 for pure dataset (all same class)', () => {
            // All samples belong to class 0
            const yValues = [
                [1, 0, 0], // class 0
                [1, 0, 0], // class 0
                [1, 0, 0], // class 0
            ];

            const result = gini.impurity(yValues);
            expect(result).toBe(0);
        });

        it('should return maximum impurity for evenly distributed three classes', () => {
            // Equal distribution among three classes
            const yValues = [
                [1, 0, 0], // class 0
                [1, 0, 0], // class 0
                [0, 1, 0], // class 1
                [0, 1, 0], // class 1
                [0, 0, 1], // class 2
                [0, 0, 1], // class 2
            ];

            const result = gini.impurity(yValues);
            // For three classes with equal distribution: Gini = 1 - (1/3² + 1/3² + 1/3²) = 1 - 3*(1/9) = 1 - 1/3 = 2/3
            expect(result).toBeCloseTo(2 / 3, 6);
        });

        it('should compute correct impurity for uneven class distribution', () => {
            // 3 samples class 0, 1 sample class 1
            const yValues = [
                [1, 0], // class 0
                [1, 0], // class 0
                [1, 0], // class 0
                [0, 1], // class 1
            ];

            const result = gini.impurity(yValues);

            // p0 = 3/4 = 0.75, p1 = 1/4 = 0.25
            // Gini = 1 - (0.75² + 0.25²) = 1 - (0.5625 + 0.0625) = 1 - 0.625 = 0.375
            expect(result).toBe(0.375);
        });

        it('should handle single sample', () => {
            const yValues = [[1, 0, 0]];

            const result = gini.impurity(yValues);
            // Single sample should have 0 impurity (pure)
            expect(result).toBe(0);
        });
    });

    describe('loss', () => {
        it('should compute loss for perfect predictions', () => {
            const yTrue = tf.tensor2d([
                [1, 0, 0], // true class 0
                [0, 1, 0], // true class 1
                [0, 0, 1], // true class 2
            ]);

            const yPred = tf.tensor2d([
                [1, 0, 0], // predicted class 0 (perfect)
                [0, 1, 0], // predicted class 1 (perfect)
                [0, 0, 1], // predicted class 2 (perfect)
            ]);

            const result = gini.loss(yTrue, yPred);
            const lossValue = result.dataSync()[0];

            // Perfect predictions should have 0 loss
            expect(lossValue).toBe(0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute loss for completely wrong predictions', () => {
            const yTrue = tf.tensor2d([
                [1, 0, 0], // true class 0
                [0, 1, 0], // true class 1
            ]);

            const yPred = tf.tensor2d([
                [0, 0, 1], // predicted class 2 (wrong)
                [1, 0, 0], // predicted class 0 (wrong)
            ]);

            const result = gini.loss(yTrue, yPred);
            const lossValue = result.dataSync()[0];

            // Wrong predictions should have higher loss
            expect(lossValue).toBeGreaterThan(0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute loss for probabilistic predictions', () => {
            const yTrue = tf.tensor2d([
                [1, 0], // true class 0
                [0, 1], // true class 1
            ]);

            const yPred = tf.tensor2d([
                [0.8, 0.2], // 80% confident class 0
                [0.3, 0.7], // 70% confident class 1
            ]);

            const result = gini.loss(yTrue, yPred);
            const lossValue = result.dataSync()[0];

            expect(lossValue).toBeGreaterThan(0);
            expect(lossValue).toBeLessThan(1);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle binary classification', () => {
            const yTrue = tf.tensor2d([
                [1, 0], // positive class
                [0, 1], // negative class
                [1, 0], // positive class
            ]);

            const yPred = tf.tensor2d([
                [0.9, 0.1], // confident positive
                [0.2, 0.8], // confident negative
                [0.6, 0.4], // less confident positive
            ]);

            const result = gini.loss(yTrue, yPred);
            const lossValue = result.dataSync()[0];

            expect(lossValue).toBeGreaterThan(0);
            expect(Number.isFinite(lossValue)).toBe(true);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('edge cases and memory management', () => {
        it('should handle very small probability values', () => {
            const yValues = [
                [0.999, 0.001],
                [0.999, 0.001],
                [0.001, 0.999],
            ];

            const result = gini.impurity(yValues);

            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThan(0);
        });

        it('should handle large number of classes', () => {
            // Create 10-class problem with some samples
            const numClasses = 10;
            const samplesPerClass = 2;

            // Create samples manually for each class
            const data: number[][] = [];
            for (let classIdx = 0; classIdx < numClasses; classIdx++) {
                for (let sample = 0; sample < samplesPerClass; sample++) {
                    const oneHot = new Array(numClasses).fill(0);
                    oneHot[classIdx] = 1;
                    data.push(oneHot);
                }
            }

            const yValues = data;

            const result = gini.impurity(yValues);

            // For equal distribution across 10 classes: Gini = 1 - 10*(1/10)² = 1 - 10/100 = 1 - 0.1 = 0.9
            expect(result).toBeCloseTo(0.9, 5);
        });

        it('should maintain consistency with mathematical definition', () => {
            // Test case where we can manually verify the calculation
            const yValues = [
                [1, 0, 0], // class 0: 4 samples
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0], // class 1: 2 samples
                [0, 1, 0],
                [0, 0, 1], // class 2: 1 sample
            ];

            const result = gini.impurity(yValues);

            // Manual calculation:
            // p0 = 4/7, p1 = 2/7, p2 = 1/7
            // Gini = 1 - (p0² + p1² + p2²)
            //      = 1 - ((4/7)² + (2/7)² + (1/7)²)
            //      = 1 - (16/49 + 4/49 + 1/49)
            //      = 1 - 21/49
            //      = 1 - 3/7
            //      = 4/7 ≈ 0.571428571...

            expect(result).toBe(4 / 7);
        });
    });
});
