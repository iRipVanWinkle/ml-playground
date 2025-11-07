import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { accuracy } from './accuracy';

describe('accuracy', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const yTrue = tf.tensor2d([[0], [1], [2]]);
            const yPred = tf.tensor2d([[0], [1], [2]]);

            const result = accuracy(yTrue, yPred);
            const accuracyValue = result.dataSync()[0];

            expect(accuracyValue).toBe(1.0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should return 0.0 for completely wrong predictions', () => {
            const yTrue = tf.tensor2d([[0], [1], [2]]);
            const yPred = tf.tensor2d([[1], [2], [0]]);

            const result = accuracy(yTrue, yPred);
            const accuracyValue = result.dataSync()[0];

            expect(accuracyValue).toBe(0.0);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should compute correct accuracy for partial matches', () => {
            const yTrue = tf.tensor2d([[0], [1], [2], [3]]);
            const yPred = tf.tensor2d([
                [0], // correct
                [1], // correct
                [0], // wrong
                [3], // correct
            ]);

            const result = accuracy(yTrue, yPred);
            const accuracyValue = result.dataSync()[0];

            // 3 out of 4 correct = 0.75
            expect(accuracyValue).toBe(0.75);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('multi-class classification', () => {
        it('should handle three-class classification', () => {
            const yTrue = tf.tensor2d([[0], [1], [2], [0], [1], [2]]);
            const yPred = tf.tensor2d([
                [0], // correct
                [1], // correct
                [2], // correct
                [1], // wrong
                [2], // wrong
                [0], // wrong
            ]);

            const result = accuracy(yTrue, yPred);
            const accuracyValue = result.dataSync()[0];

            // 3 out of 6 correct = 0.5
            expect(accuracyValue).toBe(0.5);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('memory management', () => {
        it('should not leak memory during computation', () => {
            const yTrue = tf.tensor2d([[0], [1], [2]]);
            const yPred = tf.tensor2d([[0], [1], [2]]);
            const initialTensors = tf.memory().numTensors;

            const result = accuracy(yTrue, yPred);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });

    describe('confusion matrix input', () => {
        it('should return 1.0 for perfect predictions from confusion matrix', () => {
            // Perfect predictions: all diagonal, no off-diagonal
            const confusionMatrix: number[][] = [
                [3, 0, 0], // class 0: 3 correct, 0 misclassified
                [0, 2, 0], // class 1: 2 correct, 0 misclassified
                [0, 0, 1], // class 2: 1 correct, 0 misclassified
            ];

            const result = accuracy(confusionMatrix);
            expect(result).toBe(1.0);
        });

        it('should return 0.0 for completely wrong predictions from confusion matrix', () => {
            // All predictions are wrong: no diagonal elements
            const confusionMatrix: number[][] = [
                [0, 2, 0], // class 0: all misclassified as class 1
                [0, 0, 1], // class 1: all misclassified as class 2
                [1, 0, 0], // class 2: all misclassified as class 0
            ];

            const result = accuracy(confusionMatrix);
            expect(result).toBe(0.0);
        });

        it('should compute correct accuracy for partial matches from confusion matrix', () => {
            // 3 out of 4 correct
            const confusionMatrix: number[][] = [
                [1, 0, 0], // class 0: 1 correct
                [0, 1, 0], // class 1: 1 correct
                [1, 0, 1], // class 2: 1 correct, 1 misclassified as class 0
            ];

            const result = accuracy(confusionMatrix);
            // 3 correct (diagonal) out of 4 total = 0.75
            expect(result).toBe(0.75);
        });

        it('should return 0 for empty confusion matrix', () => {
            const confusionMatrix: number[][] = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ];

            const result = accuracy(confusionMatrix);
            expect(result).toBe(0);
        });

        it('should handle binary classification confusion matrix', () => {
            // Binary classification: 8 out of 10 correct = 0.8
            const confusionMatrix: number[][] = [
                [5, 1], // class 0: 5 correct, 1 misclassified as class 1
                [1, 3], // class 1: 3 correct, 1 misclassified as class 0
            ];

            const result = accuracy(confusionMatrix);
            // 8 correct (5 + 3) out of 10 total = 0.8
            expect(result).toBe(0.8);
        });
    });
});
