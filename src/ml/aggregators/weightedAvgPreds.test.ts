import * as tf from '@tensorflow/tfjs';
import { describe, expect, it } from 'vitest';
import { weightedAvgPreds } from './weightedAvgPreds';

describe('weightedAvgPreds', () => {
    it('computes weighted averages per sample', () => {
        const predictions = tf.tensor2d([
            [1, 5, 9],
            [10, 20, 30],
        ]);
        const weights = tf.tensor2d([
            [1, 2, 1],
            [3, 1, 2],
        ]);

        const result = weightedAvgPreds(predictions, weights);
        const values = result.arraySync() as number[][];

        expect(result.shape).toEqual([2, 1]);
        expect(values[0][0]).toBeCloseTo(5, 5);
        expect(values[1][0]).toBeCloseTo(110 / 6, 5);

        predictions.dispose();
        weights.dispose();
        result.dispose();
    });

    it('supports zero-valued weights by clamping the denominator', () => {
        const predictions = tf.tensor2d([[1, 2, 3]]);
        const weights = tf.tensor2d([[0, 0, 0]]);

        const result = weightedAvgPreds(predictions, weights);
        const values = result.arraySync() as number[][];

        expect(values[0][0]).toBeCloseTo(0, 5);

        predictions.dispose();
        weights.dispose();
        result.dispose();
    });

    it('throws for mismatched shapes', () => {
        const predictions = tf.tensor2d([[1, 2]]);
        const weights = tf.tensor2d([[1, 2, 3]]);

        expect(() => weightedAvgPreds(predictions, weights)).toThrow(
            'Predictions and weights must have the same shape',
        );

        predictions.dispose();
        weights.dispose();
    });
});
