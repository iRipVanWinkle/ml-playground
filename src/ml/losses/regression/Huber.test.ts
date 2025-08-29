import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { Huber } from './Huber';

describe('Huber', () => {
    describe('compute', () => {
        it('returns 0 for identical predictions and true values', () => {
            const loss = new Huber();
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1], [2], [3]]);

            const result = loss.compute(yTrue, yPred).arraySync();

            expect(result).toBe(0);
        });

        it('returns correct loss for slightly different predictions', () => {
            const loss = new Huber(1.0);
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1.1], [1.9], [3.2]]);

            const result = loss.compute(yTrue, yPred).arraySync();

            expect(result).toBeCloseTo(0.01);
        });

        it('should keep memory clear', () => {
            const loss = new Huber();
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1], [2], [3]]);

            const prevNumTensors = tf.memory().numTensors;

            loss.compute(yTrue, yPred);

            const expectedNumTensors = prevNumTensors + 1;

            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });

    describe('parameterGradient', () => {
        it('computes correct gradient for simple case', () => {
            const loss = new Huber(1.0);

            const xTrue = tf.tensor2d([
                [1, 2],
                [2, 3],
                [3, 4],
            ]);
            const yTrue = tf.tensor2d([[1], [2], [4]]);
            const yPred = tf.tensor2d([[0], [0], [0]]);

            const gradient = loss.parameterGradient(xTrue, yTrue, yPred);

            expect(gradient.arraySync()[0]).toBeCloseTo(-3, 2);
            expect(gradient.arraySync()[1]).toBeCloseTo(-6, 2);
            expect(gradient.arraySync()[2]).toBeCloseTo(-9, 2);
        });

        it('should keep memory clear', () => {
            const loss = new Huber();

            const xTrue = tf.tensor2d([
                [1, 2],
                [2, 3],
                [3, 4],
            ]);
            const yTrue = tf.tensor2d([[1], [2], [4]]);
            const yPred = tf.tensor2d([[0], [0], [0]]);

            const prevNumTensors = tf.memory().numTensors;

            loss.parameterGradient(xTrue, yTrue, yPred);

            const expectedNumTensors = prevNumTensors + 1;

            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });
});
