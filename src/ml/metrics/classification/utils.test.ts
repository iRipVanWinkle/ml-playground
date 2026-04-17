import { describe, it, expect } from 'vitest';
import { macroAverage, weightedAverage } from './utils';

describe('macroAverage', () => {
    it('should return 0 for an empty array', () => {
        expect(macroAverage([])).toBe(0);
    });

    it('should return the correct average for an array of numbers', () => {
        expect(macroAverage([0.8, 0.9])).toBeCloseTo(0.85);
        expect(macroAverage([1, 0.5, 0])).toBeCloseTo(0.5);
        expect(macroAverage([0.7, 0.7, 0.7])).toBeCloseTo(0.7);
    });
});

describe('weightedAverage', () => {
    const values = [0.8, 0.9];
    const confusionMatrix = [
        [10, 2], // row sum: 12, col sum: 13
        [3, 15], // row sum: 18, col sum: 17
    ];

    it('should return 0 if total weight is 0', () => {
        const emptyMatrix = [
            [0, 0],
            [0, 0],
        ];
        expect(weightedAverage(values, emptyMatrix)).toBe(0);
    });

    it('should calculate the weighted average using row sums by default', () => {
        // (0.8 * 12 + 0.9 * 18) / 30 = (9.6 + 16.2) / 30 = 25.8 / 30 = 0.86
        expect(weightedAverage(values, confusionMatrix)).toBeCloseTo(0.86);
    });

    it('should calculate the weighted average using column sums when useRowSums is false', () => {
        // (0.8 * 13 + 0.9 * 17) / 30 = (10.4 + 15.3) / 30 = 25.7 / 30 = 0.85666...
        expect(weightedAverage(values, confusionMatrix, false)).toBeCloseTo(0.8566666666666667);
    });
});
