import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { LogScaler } from './logScaling';

describe('logScaling', () => {
    it('returns empty tensor for empty matrix', () => {
        const input = tf.tensor2d([], [0, 0]);
        const scaler = new LogScaler();
        scaler.fit();
        const result = scaler.transform(input);
        expect(result.shape).toEqual([0, 0]);
        expect(result.arraySync()).toEqual([]);
    });

    it('returns empty tensor for matrix with empty row', () => {
        const input = tf.tensor2d([[]]);
        const scaler = new LogScaler();
        scaler.fit();
        const result = scaler.transform(input);
        expect(result.shape).toEqual([0, 0]);
        expect(result.arraySync()).toEqual([]);
    });

    it('throws error for matrix with zero', () => {
        const scaler = new LogScaler();
        scaler.fit();
        expect(() => scaler.transform(tf.tensor2d([[0]]))).toThrow(
            `Log scaling requires all values to be positive. Found minimum value: 0`,
        );
        expect(() =>
            scaler.transform(
                tf.tensor2d([
                    [1, 2],
                    [3, 0],
                ]),
            ),
        ).toThrow(`Log scaling requires all values to be positive. Found minimum value: 0`);
    });

    it('throws error for matrix with negative value', () => {
        const scaler = new LogScaler();
        scaler.fit();
        expect(() => scaler.transform(tf.tensor2d([[-1]]))).toThrow(
            `Log scaling requires all values to be positive. Found minimum value: -1`,
        );
        expect(() =>
            scaler.transform(
                tf.tensor2d([
                    [1, 2],
                    [3, -5],
                ]),
            ),
        ).toThrow(`Log scaling requires all values to be positive. Found minimum value: -5`);
    });

    it('scales a matrix with one positive element', () => {
        const input = tf.tensor2d([[5]]);
        const expected = tf.tensor2d([[Math.log(5)]]);
        const scaler = new LogScaler();
        scaler.fit();
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with all positive elements', () => {
        const input = tf.tensor2d([
            [1, 2],
            [3, 4],
        ]);
        const expected = tf.tensor2d([
            [Math.log(1), Math.log(2)],
            [Math.log(3), Math.log(4)],
        ]);
        const scaler = new LogScaler();
        scaler.fit();
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with multiple rows and columns', () => {
        const input = tf.tensor2d([
            [2, 4, 8],
            [16, 32, 64],
        ]);
        const expected = tf.tensor2d([
            [Math.log(2), Math.log(4), Math.log(8)],
            [Math.log(16), Math.log(32), Math.log(64)],
        ]);
        const scaler = new LogScaler();
        scaler.fit();
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('scales a matrix with large positive numbers', () => {
        const input = tf.tensor2d([
            [1000, 10000],
            [100000, 1000000],
        ]);
        const expected = tf.tensor2d([
            [Math.log(1000), Math.log(10000)],
            [Math.log(100000), Math.log(1000000)],
        ]);
        const scaler = new LogScaler();
        scaler.fit();
        const result = scaler.transform(input);
        expect(result.arraySync()).toEqual(expected.arraySync());
    });

    it('does not mutate the input tensor', () => {
        const input = tf.tensor2d([
            [1, 2],
            [3, 4],
        ]);
        const inputCopy = input.clone();
        const scaler = new LogScaler();
        scaler.fit();
        scaler.transform(input);
        expect(input.arraySync()).toEqual(inputCopy.arraySync());
    });
});
