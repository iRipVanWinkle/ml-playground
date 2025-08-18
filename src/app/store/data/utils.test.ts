import { describe, expect, it } from 'vitest';
import { generateCartesianProduct } from './utils';

describe('generateCartesianProduct', () => {
    it('should generate Cartesian product for 1 column', () => {
        const predictionsNum = 3;
        const xMin = [0];
        const xMax = [2];

        const result = generateCartesianProduct(predictionsNum, xMin, xMax);

        expect(result).toEqual([[0], [1], [2]]);
    });

    it('should generate Cartesian product for 2 columns', () => {
        const predictionsNum = 2;
        const xMin = [0, 0];
        const xMax = [1, 1];

        const result = generateCartesianProduct(predictionsNum, xMin, xMax);

        expect(result).toEqual([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]);
    });

    it('should generate Cartesian product for 3 columns', () => {
        const predictionsNum = 2;
        const xMin = [0, 0, 0];
        const xMax = [1, 1, 1];

        const result = generateCartesianProduct(predictionsNum, xMin, xMax);

        expect(result).toEqual([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]);
    });

    it('should handle empty input', () => {
        const predictionsNum = 0;

        const result = generateCartesianProduct(predictionsNum, [], []);

        expect(result).toEqual([[]]);
    });
});
