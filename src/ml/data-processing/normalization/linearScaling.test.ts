import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { linearScaling } from './linearScaling';

describe('linearScaling', () => {
    it('returns empty tensor for empty matrix', () => {
        const input = tf.tensor2d([], [0, 0]);
        const result = linearScaling(input);
        expect(result.shape).toEqual([0, 0]);
        expect(result.arraySync()).toEqual([]);
    });

    it('returns empty tensor for matrix with empty row', () => {
        const input = tf.tensor2d([[]]);
        const result = linearScaling(input);
        expect(result.shape).toEqual([0, 0]);
        expect(result.arraySync()).toEqual([]);
    });

    it('scales a matrix with one element to [[0]]', () => {
        const input = tf.tensor2d([[5]]);
        const expected = tf.tensor2d([[0]]);
        const result = linearScaling(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with all elements the same to all 0', () => {
        const input = tf.tensor2d([
            [2, 2],
            [2, 2],
        ]);
        const expected = tf.tensor2d([
            [0, 0],
            [0, 0],
        ]);
        const result = linearScaling(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with positive numbers', () => {
        const input = tf.tensor2d([
            [1, 2],
            [3, 4],
        ]);
        const expected = tf.tensor2d([
            [0, 0.3333333333333333],
            [0.6666666666666666, 1],
        ]);
        const result = linearScaling(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with negative numbers', () => {
        const input = tf.tensor2d([
            [-4, -2],
            [-3, -1],
        ]);
        const expected = tf.tensor2d([
            [0, 0.6666666666666666],
            [0.3333333333333333, 1],
        ]);
        const result = linearScaling(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with mixed positive and negative numbers', () => {
        const input = tf.tensor2d([
            [-2, 0],
            [2, 4],
        ]);
        const expected = tf.tensor2d([
            [0, 0.3333333333333333],
            [0.6666666666666666, 1],
        ]);
        const result = linearScaling(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with multiple rows and columns', () => {
        const input = tf.tensor2d([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        const expected = tf.tensor2d([
            [0, 0.2, 0.4],
            [0.6, 0.8, 1],
        ]);
        const result = linearScaling(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });
});
