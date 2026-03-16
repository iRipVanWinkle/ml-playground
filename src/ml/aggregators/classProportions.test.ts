import * as tf from '@tensorflow/tfjs';
import { describe, expect, it } from 'vitest';
import { classProportions } from './classProportions';

describe('classProportions', () => {
    it('returns uniform vote proportions per sample', () => {
        const labels = tf.tensor2d([
            [0, 1, 1],
            [2, 2, 1],
        ]);

        const result = classProportions(labels, 3);
        const values = result.arraySync() as number[][];

        expect(result.shape).toEqual([2, 3]);
        expect(values[0][0]).toBeCloseTo(1 / 3, 5);
        expect(values[0][1]).toBeCloseTo(2 / 3, 5);
        expect(values[0][2]).toBeCloseTo(0, 5);
        expect(values[1][0]).toBeCloseTo(0, 5);
        expect(values[1][1]).toBeCloseTo(1 / 3, 5);
        expect(values[1][2]).toBeCloseTo(2 / 3, 5);

        labels.dispose();
        result.dispose();
    });

    it('throws when numClasses is invalid', () => {
        const labels = tf.tensor2d([[0, 1]]);

        expect(() => classProportions(labels, 0)).toThrow('numClasses must be a positive integer');

        labels.dispose();
    });
});
