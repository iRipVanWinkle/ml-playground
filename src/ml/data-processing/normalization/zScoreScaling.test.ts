import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { zScoreScaling } from './zScoreScaling';

describe('zScoreScaling', () => {
    it('returns [] for empty matrix', () => {
        expect(zScoreScaling(tf.tensor2d([], [0, 0])).arraySync()).toEqual([]);
    });

    it('returns [] for matrix with empty row', () => {
        expect(zScoreScaling(tf.tensor2d([[]])).arraySync()).toEqual([]);
    });

    it('returns [[0]] for matrix with one element', () => {
        expect(zScoreScaling(tf.tensor2d([[5]])).arraySync()).toEqual([[0]]); // 5 - 5 / 1e-8 = 500000000
    });

    it('returns matrix of 0s for all elements the same', () => {
        const input = [
            [2, 2],
            [2, 2],
        ];
        const result = zScoreScaling(tf.tensor2d(input)).arraySync();
        expect(result).toEqual([
            [0, 0],
            [0, 0],
        ]);
    });

    // it('scales a matrix with negative values', () => {
    //   const matrix = [
    //     [-3, -2],
    //     [-1, 0],
    //   ];
    //   const result = zScoreScaling(tf.tensor2d(matrix)).arraySync();
    //   const flat = matrix.flat();
    //   const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    //   const std = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
    //   const expected = matrix.map(row => row.map(val => (val - mean) / std));
    //   expect(result[0][0]).toBeCloseTo(expected[0][0]);
    //   expect(result[0][1]).toBeCloseTo(expected[0][1]);
    //   expect(result[1][0]).toBeCloseTo(expected[1][0]);
    //   expect(result[1][1]).toBeCloseTo(expected[1][1]);
    // });

    // it('scales a matrix with both negative and positive values', () => {
    //   const matrix = [
    //     [-2, 0],
    //     [2, 4],
    //   ];
    //   const result = zScoreScaling(tf.tensor2d(matrix)).arraySync();
    //   const flat = matrix.flat();
    //   const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    //   const std = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
    //   const expected = matrix.map(row => row.map(val => (val - mean) / std));
    //   expect(result[0][0]).toBeCloseTo(expected[0][0]);
    //   expect(result[0][1]).toBeCloseTo(expected[0][1]);
    //   expect(result[1][0]).toBeCloseTo(expected[1][0]);
    //   expect(result[1][1]).toBeCloseTo(expected[1][1]);
    // });

    // it('scales a non-square matrix', () => {
    //   const matrix = [
    //     [1, 2, 3],
    //     [4, 5, 6],
    //   ];
    //   const result = zScoreScaling(tf.tensor2d(matrix)).arraySync();
    //   const flat = matrix.flat();
    //   const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    //   const std = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
    //   const expected = matrix.map(row => row.map(val => (val - mean) / std));
    //   expect(result[0][0]).toBeCloseTo(expected[0][0]);
    //   expect(result[0][1]).toBeCloseTo(expected[0][1]);
    //   expect(result[0][2]).toBeCloseTo(expected[0][2]);
    //   expect(result[1][0]).toBeCloseTo(expected[1][0]);
    //   expect(result[1][1]).toBeCloseTo(expected[1][1]);
    //   expect(result[1][2]).toBeCloseTo(expected[1][2]);
    // });
});
