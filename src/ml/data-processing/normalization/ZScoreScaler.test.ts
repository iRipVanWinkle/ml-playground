import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { ZScoreScaler } from './ZScoreScaler';

describe('zScoreScaling', () => {
    it('throws error for empty matrix', () => {
        const scaler = new ZScoreScaler();
        expect(() => scaler.fit(tf.tensor2d([], [0, 0]))).toThrow('Input tensor is empty');
        expect(() => scaler.transform(tf.tensor2d([], [0, 0]))).toThrow(
            'Scaler has not been fitted yet',
        );
    });

    it('throws error for matrix with empty row', () => {
        const scaler = new ZScoreScaler();
        expect(() => scaler.fit(tf.tensor2d([[]]))).toThrow('Input tensor is empty');
    });

    it('returns [[0]] for matrix with one element', () => {
        const input = tf.tensor2d([[5]]);
        const scaler = new ZScoreScaler();
        scaler.fit(input);
        expect(scaler.transform(input).arraySync()).toEqual([[0]]);
    });

    it('returns matrix of 0s for all elements the same', () => {
        const input = [
            [2, 2],
            [2, 2],
        ];
        const scaler = new ZScoreScaler();
        const inputT = tf.tensor2d(input);
        scaler.fit(inputT);
        const result = scaler.transform(inputT).arraySync();
        expect(result).toEqual([
            [0, 0],
            [0, 0],
        ]);
    });
});
