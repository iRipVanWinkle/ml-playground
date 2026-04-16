import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { MinMaxScaler } from './MinMaxScaler';

describe('linearScaling', () => {
    it('throws error for empty matrix', () => {
        const input = tf.tensor2d([], [0, 0]);
        const scaler = new MinMaxScaler();
        expect(() => scaler.fit(input)).toThrow('Input tensor is empty');
        expect(() => scaler.transform(input)).toThrow('Scaler has not been fitted yet');
    });

    it('throws error for matrix with empty row', () => {
        const input = tf.tensor2d([[]]);
        const scaler = new MinMaxScaler();
        expect(() => scaler.fit(input)).toThrow('Input tensor is empty');
    });

    it('scales a matrix with one element to [[0]]', () => {
        const input = tf.tensor2d([[5]]);
        const expected = tf.tensor2d([[0]]);
        const scaler = new MinMaxScaler();
        scaler.fit(input);
        const result = scaler.transform(input);
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
        const scaler = new MinMaxScaler();
        scaler.fit(input);
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with positive numbers per feature', () => {
        const input = tf.tensor2d([
            [1, 2],
            [3, 6],
            [5, 4],
        ]);
        // Col 0: [1, 3, 5] -> Min: 1, Max: 5, Range: 4 -> [0, 0.5, 1]
        // Col 1: [2, 6, 4] -> Min: 2, Max: 6, Range: 4 -> [0, 1, 0.5]
        const expected = tf.tensor2d([
            [0, 0],
            [0.5, 1],
            [1, 0.5],
        ]);
        const scaler = new MinMaxScaler();
        scaler.fit(input);
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with negative numbers per feature', () => {
        const input = tf.tensor2d([
            [-4, -2],
            [-2, -6],
            [0, -4],
        ]);
        // Col 0: [-4, -2, 0] -> Min: -4, Max: 0, Range: 4 -> [0, 0.5, 1]
        // Col 1: [-2, -6, -4] -> Min: -6, Max: -2, Range: 4 -> [1, 0, 0.5]
        const expected = tf.tensor2d([
            [0, 1],
            [0.5, 0],
            [1, 0.5],
        ]);
        const scaler = new MinMaxScaler();
        scaler.fit(input);
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with mixed positive and negative numbers per feature', () => {
        const input = tf.tensor2d([
            [-2, 0],
            [0, 4],
            [2, -4],
        ]);
        // Col 0: [-2, 0, 2] -> Min: -2, Max: 2, Range: 4 -> [0, 0.5, 1]
        // Col 1: [0, 4, -4] -> Min: -4, Max: 4, Range: 8 -> [0.5, 1, 0]
        const expected = tf.tensor2d([
            [0, 0.5],
            [0.5, 1],
            [1, 0],
        ]);
        const scaler = new MinMaxScaler();
        scaler.fit(input);
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with multiple rows and columns per feature', () => {
        const input = tf.tensor2d([
            [1, 5, 3],
            [4, 2, 6],
            [7, 8, 9],
        ]);
        // Col 0: [1, 4, 7] -> Min 1, Max 7, Range 6 -> [0, 0.5, 1]
        // Col 1: [5, 2, 8] -> Min 2, Max 8, Range 6 -> [0.5, 0, 1]
        // Col 2: [3, 6, 9] -> Min 3, Max 9, Range 6 -> [0, 0.5, 1]
        const expected = tf.tensor2d([
            [0, 0.5, 0],
            [0.5, 0, 0.5],
            [1, 1, 1],
        ]);
        const scaler = new MinMaxScaler();
        scaler.fit(input);
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });
});
