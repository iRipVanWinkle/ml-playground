import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';
import { confusionMatrix } from './confusionMatrix';

describe('confusionMatrix', () => {
    describe('basic functionality', () => {
        it('should create identity matrix for perfect predictions', () => {
            const yTrue = tf.tensor2d([[0], [1], [2]]);
            const yPred = tf.tensor2d([[0], [1], [2]]);
            const numClasses = 3;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync();

            // Should be identity matrix
            expect(cm).toEqual([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should create correct confusion matrix for binary classification', () => {
            const yTrue = tf.tensor2d([[0], [0], [1], [1]]);
            const yPred = tf.tensor2d([
                [0], // correct
                [1], // wrong (predicted 1, actual 0)
                [1], // correct
                [0], // wrong (predicted 0, actual 1)
            ]);
            const numClasses = 2;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync();

            // Confusion matrix:
            //         Predicted
            //         0    1
            // Actual 0 [1, 1]  (1 correct, 1 wrong)
            //        1 [1, 1]  (1 wrong, 1 correct)
            expect(cm).toEqual([
                [1, 1],
                [1, 1],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle all wrong predictions', () => {
            const yTrue = tf.tensor2d([[0], [1]]);
            const yPred = tf.tensor2d([[1], [0]]);
            const numClasses = 2;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync();

            expect(cm).toEqual([
                [0, 1],
                [1, 0],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('multi-class classification', () => {
        it('should create correct confusion matrix for three classes', () => {
            const yTrue = tf.tensor2d([[0], [0], [1], [1], [2], [2]]);
            const yPred = tf.tensor2d([
                [0], // correct
                [1], // wrong: predicted 1, actual 0
                [1], // correct
                [2], // wrong: predicted 2, actual 1
                [2], // correct
                [0], // wrong: predicted 0, actual 2
            ]);
            const numClasses = 3;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync();

            // Confusion matrix:
            //         Predicted
            //         0    1    2
            // Actual 0 [1,  1,  0]
            //        1 [0,  1,  1]
            //        2 [1,  0,  1]
            expect(cm).toEqual([
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle uneven class distribution', () => {
            const yTrue = tf.tensor2d([[0], [0], [0], [1], [2]]);
            const yPred = tf.tensor2d([
                [0],
                [0],
                [1], // wrong: predicted 1, actual 0
                [1],
                [2],
            ]);
            const numClasses = 3;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync();

            // Class 0: 2 correct, 1 predicted as class 1
            // Class 1: 1 correct
            // Class 2: 1 correct
            expect(cm).toEqual([
                [2, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle single sample', () => {
            const yTrue = tf.tensor2d([[1]]);
            const yPred = tf.tensor2d([[1]]);
            const numClasses = 3;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync();

            expect(cm).toEqual([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });

        it('should handle all samples in one class', () => {
            const yTrue = tf.tensor2d([[0], [0], [0]]);
            const yPred = tf.tensor2d([[0], [0], [0]]);
            const numClasses = 3;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            const cm = result.arraySync() as number[][];

            expect(cm).toEqual([
                [3, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]);

            yTrue.dispose();
            yPred.dispose();
            result.dispose();
        });
    });

    describe('memory management', () => {
        it('should not leak memory during computation', () => {
            const yTrue = tf.tensor2d([[0], [1], [2]]);
            const yPred = tf.tensor2d([[0], [1], [2]]);
            const numClasses = 3;

            const initialTensors = tf.memory().numTensors;

            const result = confusionMatrix(yTrue, yPred, numClasses);
            result.dispose();

            const finalTensors = tf.memory().numTensors;
            expect(finalTensors).toBe(initialTensors);

            yTrue.dispose();
            yPred.dispose();
        });
    });
});
