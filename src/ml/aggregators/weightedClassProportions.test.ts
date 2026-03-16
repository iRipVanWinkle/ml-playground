import * as tf from '@tensorflow/tfjs';
import { describe, expect, it } from 'vitest';
import { weightedClassProportions } from './weightedClassProportions';

describe('weightedClassProportions', () => {
    it('returns inverse-style weighted class proportions per sample', () => {
        const labels = tf.tensor2d([
            [0, 1, 1],
            [2, 2, 1],
        ]);
        const weights = tf.tensor2d([
            [4, 1, 1],
            [1, 3, 2],
        ]);

        const result = weightedClassProportions(labels, weights, 3);
        const values = result.arraySync() as number[][];

        expect(result.shape).toEqual([2, 3]);
        expect(values[0][0]).toBeCloseTo(4 / 6, 5);
        expect(values[0][1]).toBeCloseTo(2 / 6, 5);
        expect(values[0][2]).toBeCloseTo(0, 5);
        expect(values[1][0]).toBeCloseTo(0, 5);
        expect(values[1][1]).toBeCloseTo(2 / 6, 5);
        expect(values[1][2]).toBeCloseTo(4 / 6, 5);

        labels.dispose();
        weights.dispose();
        result.dispose();
    });

    it('throws for mismatched label and weight shapes', () => {
        const labels = tf.tensor2d([[0, 1]]);
        const weights = tf.tensor2d([[1, 2, 3]]);

        expect(() => weightedClassProportions(labels, weights, 2)).toThrow(
            'Labels and weights must have the same shape',
        );

        labels.dispose();
        weights.dispose();
    });
});
