import { describe, it, expect, beforeEach } from 'vitest';
import { MeanAbsoluteError } from './MeanAbsoluteError';

describe('MeanAbsoluteError', () => {
    let mae: MeanAbsoluteError;

    beforeEach(() => {
        mae = new MeanAbsoluteError();
    });

    describe('impurity', () => {
        it('should return 0 for identical values', () => {
            const yTrue = [[5], [5], [5], [5]];

            const result = mae.impurity(yTrue);
            expect(result).toBeCloseTo(0, 6);
        });

        it('should compute correct MAD for simple case', () => {
            // Values: [1, 2, 3, 4, 5] -> median = 3
            // MAD = (|1-3| + |2-3| + |3-3| + |4-3| + |5-3|) / 5 = (2 + 1 + 0 + 1 + 2) / 5 = 1.2
            const yTrue = [[1], [2], [3], [4], [5]];

            const result = mae.impurity(yTrue);
            expect(result).toBeCloseTo(1.2, 6);
        });

        it('should compute correct MAD for even number of samples', () => {
            // Values: [1, 2, 4, 5] -> median = (2 + 4) / 2 = 3
            // MAD = (|1-3| + |2-3| + |4-3| + |5-3|) / 4 = (2 + 1 + 1 + 2) / 4 = 1.5
            const yTrue = [[1], [2], [4], [5]];

            const result = mae.impurity(yTrue);
            expect(result).toBeCloseTo(1.5, 6);
        });

        it('should handle single value', () => {
            const yTrue = [[42]];

            const result = mae.impurity(yTrue);
            expect(result).toBeCloseTo(0, 6);
        });

        it('should be robust to outliers', () => {
            // Compare to MSE: outliers should have less impact
            const yTrueWithOutlier = [[1], [2], [3], [100]];

            const result = mae.impurity(yTrueWithOutlier);

            // Median of [1, 2, 3, 100] = (2 + 3) / 2 = 2.5
            // MAD = (|1-2.5| + |2-2.5| + |3-2.5| + |100-2.5|) / 4 = (1.5 + 0.5 + 0.5 + 97.5) / 4 = 25
            expect(result).toBeCloseTo(25, 5);
        });

        it('should handle negative values', () => {
            const yTrue = [[-5], [-1], [0], [1], [5]];

            const result = mae.impurity(yTrue);

            // Median = 0, MAD = (5 + 1 + 0 + 1 + 5) / 5 = 2.4
            expect(result).toBeCloseTo(2.4, 6);
        });
    });
});
