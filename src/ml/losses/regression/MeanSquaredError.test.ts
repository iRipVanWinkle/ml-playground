import { describe, it, expect, afterEach, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { MeanSquaredError } from './MeanSquaredError';
import type { LossFunction } from '../../types';

describe('MeanSquaredError', () => {
    let loss: LossFunction;

    afterEach(() => {
        tf.engine().disposeVariables();
        tf.disposeVariables();
        tf.dispose();
    });

    beforeEach(() => {
        loss = new MeanSquaredError();
    });

    describe('compute', () => {
        it('returns 0 for identical predictions and true values', () => {
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1], [2], [3]]);

            const result = loss.compute(yTrue, yPred).arraySync();

            expect(result).toBe(0);
        });

        it('returns correct MSE for simple case', () => {
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1.1], [1.9], [3.2]]);

            const result = loss.compute(yTrue, yPred).arraySync();

            expect(result).toBeCloseTo(0.02);
        });

        it('should keep memory clear', () => {
            const yTrue = tf.tensor2d([[1], [2], [3]]);
            const yPred = tf.tensor2d([[1.1], [1.9], [3.2]]);

            const prevNumTensors = tf.memory().numTensors;

            loss.compute(yTrue, yPred).arraySync();

            const expectedNumTensors = prevNumTensors + 1;

            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });

    describe('parameterGradient', () => {
        it('computes correct gradient for simple case', () => {
            const xTrue = tf.tensor2d([
                [1, 2],
                [2, 3],
                [3, 4],
            ]);
            const yTrue = tf.tensor2d([[1], [2], [4]]);
            const yPred = tf.tensor2d([[0], [0], [0]]);

            const gradient = loss.parameterGradient(xTrue, yTrue, yPred);

            expect(gradient.arraySync()[0]).toBeCloseTo(-2.33, 2);
            expect(gradient.arraySync()[1]).toBeCloseTo(-5.67, 2);
            expect(gradient.arraySync()[2]).toBeCloseTo(-8, 2);
        });

        it('should keep memory clear', () => {
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
