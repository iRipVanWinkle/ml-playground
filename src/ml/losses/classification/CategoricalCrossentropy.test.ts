import { describe, it, expect, afterEach, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { CategoricalCrossentropy } from './CategoricalCrossentropy';

const round = (array: number[][]) => {
    return array.map((row) => row.map((n) => Number(n.toFixed(2))));
};

describe('CategoricalCrossentropy', () => {
    let loss: LossFunction;

    afterEach(() => {
        tf.engine().disposeVariables();
        tf.disposeVariables();
        tf.dispose();
    });

    beforeEach(() => {
        loss = new CategoricalCrossentropy();
    });

    describe('::compute', () => {
        it('should compute loss for 3 classes', () => {
            const yTrue = tf.tensor2d([
                [1, 0, 0], // class 0
                [0, 1, 0], // class 1
                [0, 0, 1], // class 2
            ]);
            const yPred = tf.tensor2d([
                [0.9, 0.05, 0.05], // predicted mostly class 0 → good
                [0.1, 0.8, 0.1], // predicted mostly class 1 → good
                [0.2, 0.2, 0.6], // predicted mostly class 2 → good
            ]);

            const result = loss.compute(yTrue, yPred);

            expect(result.dataSync()[0]).toBeCloseTo(0.2798, 4);
        });

        it('should compute loss for 4 classes', () => {
            const yTrue = tf.tensor2d([
                [1, 0, 0], // true class: 0
                [0, 1, 0], // true class: 1
                [0, 0, 1], // true class: 2
                [0, 1, 0], // true class: 1
            ]);
            const yPred = tf.tensor2d([
                [0.6, 0.2, 0.2], // mostly class 0 → ok
                [0.2, 0.5, 0.3], // correct but less confident
                [0.3, 0.3, 0.4], // low confidence correct
                [0.7, 0.2, 0.1], // wrong! class 0 predicted instead of class 1
            ]);
            const result = loss.compute(yTrue, yPred);

            expect(result.dataSync()[0]).toBeCloseTo(0.9324, 4);
        });
    });

    describe('::parameterGradient', () => {
        it('should compute parameter gradient for 4 classes', () => {
            const X = tf.tensor2d([
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 0.5],
                [3.0, 1.0, 0.0],
                [1.0, 0.0, 2.0],
                [0.5, 0.5, 0.5],
            ]);

            const yTrue = tf.tensor2d([
                [1, 0, 0, 0], // class 0
                [0, 1, 0, 0], // class 1
                [0, 0, 1, 0], // class 2
                [0, 0, 0, 1], // class 3
                [0, 1, 0, 0], // class 1
            ]);

            const yPred = tf.tensor2d([
                [0.6, 0.1, 0.2, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.2, 0.5, 0.1],
                [0.1, 0.1, 0.2, 0.6],
                [0.3, 0.4, 0.2, 0.1],
            ]);

            const result = loss.parameterGradient(X, yTrue, yPred);

            const expectedGrad = [
                [0.06, -0.1, 0.04, 0], // Bias gradients
                [0.09, 0.1, -0.2, 0.01], // Weight gradients for feature 1
                [-0.07, -0.04, 0.02, 0.09], // Weight gradients for feature 2
                [-0.16, 0.01, 0.23, -0.08], // Weight gradients for feature 3
            ];

            expect(round(result.arraySync())).toEqual(expectedGrad);
        });
    });
});
