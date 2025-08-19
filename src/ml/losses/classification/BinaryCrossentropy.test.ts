import { describe, it, expect, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { BinaryCrossentropy } from './BinaryCrossentropy';

describe('BinaryCrossentropy', () => {
    let loss: LossFunction;

    beforeEach(() => {
        loss = new BinaryCrossentropy();
    });

    describe('::compute', () => {
        it('returns identical predictions and true values', () => {
            const yTrue = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);
            const yPred = tf.tensor2d([[0.2689], [0.2689], [0.2689], [0.6225], [0.7311], [0.6225]]);

            const result = loss.compute(yTrue, yPred).arraySync();

            expect(result).toBeCloseTo(0.3668, 4);
        });

        it('returns identical predictions and true values', () => {
            const yTrue = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);
            const yPred = tf.tensor2d([[0.1192], [0.1192], [0.1192], [0.3775], [0.5], [0.3775]]);

            const result = loss.compute(yTrue, yPred).arraySync();

            expect(result).toBeCloseTo(0.5037, 4);
        });

        it('should keep memory clear', () => {
            const yTrue = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);
            const yPred = tf.tensor2d([[0.1192], [0.1192], [0.1192], [0.3775], [0.5], [0.3775]]);

            const prevNumTensors = tf.memory().numTensors;

            loss.compute(yTrue, yPred).arraySync();

            const expectedNumTensors = prevNumTensors + 1;

            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });

    describe('::parameterGradient', () => {
        it('computes correct gradient for simple case', () => {
            const xTrue = tf.tensor2d([
                [0.5, 1.5],
                [1, 1],
                [1.5, 0.5],
                [3, 0.5],
                [2, 2],
                [1, 2.5],
            ]);
            const yTrue = tf.tensor2d([[0], [0], [0], [1], [1], [1]]);
            const yPred = tf.tensor2d([
                [0.9985],
                [0.9975],
                [0.9959],
                [0.9998],
                [0.99998],
                [0.99997],
            ]);

            const gradient = loss.parameterGradient(xTrue, yTrue, yPred);

            expect(gradient.arraySync()[0]).toBeCloseTo(0.4986, 4);
            expect(gradient.arraySync()[1]).toBeCloseTo(0.4983, 4);
            expect(gradient.arraySync()[2]).toBeCloseTo(0.4988, 4);
        });
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
