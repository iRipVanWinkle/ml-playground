import { describe, it, expect, beforeEach } from 'vitest';
import { MeanSquaredError } from './MeanSquaredError';

describe('MeanSquaredError', () => {
    let mse: MeanSquaredError;

    beforeEach(() => {
        mse = new MeanSquaredError();
    });

    describe('impurity', () => {
        it('should return 0 for identical values', () => {
            const yTrue = [[5], [5], [5], [5]];

            const result = mse.impurity(yTrue);

            expect(result).toBeCloseTo(0, 6);
        });

        it('should compute correct variance for simple case', () => {
            // Values: [1, 2, 3, 4, 5] -> mean = 3
            // Variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = (4 + 1 + 0 + 1 + 4) / 5 = 2
            const yTrue = [[1], [2], [3], [4], [5]];

            const result = mse.impurity(yTrue);

            expect(result).toBeCloseTo(2, 6);
        });

        it('should handle single value', () => {
            const yTrue = [[42]];

            const result = mse.impurity(yTrue);

            expect(result).toBeCloseTo(0, 6);
        });

        it('should be sensitive to outliers', () => {
            // Values: [1, 2, 3, 100] -> mean = 26.5
            // Variance = ((1-26.5)² + (2-26.5)² + (3-26.5)² + (100-26.5)²) / 4
            //          = (650.25 + 600.25 + 552.25 + 5402.25) / 4 = 7205 / 4 = 1801.25
            const yTrueWithOutlier = [[1], [2], [3], [100]];

            const result = mse.impurity(yTrueWithOutlier);

            expect(result).toBeCloseTo(1801.25, 2);
        });

        it('should handle negative values', () => {
            // Values: [-2, -1, 0, 1, 2] -> mean = 0
            // Variance = (4 + 1 + 0 + 1 + 4) / 5 = 2
            const yTrue = [[-2], [-1], [0], [1], [2]];

            const result = mse.impurity(yTrue);

            expect(result).toBeCloseTo(2, 6);
        });

        it('should handle large datasets efficiently', () => {
            // Create a larger dataset
            const size = 1000;
            const data: number[][] = [];
            for (let i = 0; i < size; i++) {
                data.push([i]);
            }
            const yTrue = data;

            const result = mse.impurity(yTrue);

            // For sequence 0, 1, 2, ..., 999: mean = 499.5
            // Variance = sum((i - 499.5)²) / 1000
            // This should be approximately (1000² - 1) / 12 ≈ 83333.25
            expect(result).toBeCloseTo(83333.25, 0);
        });

        it('should handle very small values', () => {
            const yTrue = [[0.001], [0.002], [0.003]];

            const result = mse.impurity(yTrue);

            // Mean = 0.002, variance = ((0.001-0.002)² + (0.002-0.002)² + (0.003-0.002)²) / 3
            //                        = (0.000001 + 0 + 0.000001) / 3 = 0.000002 / 3 ≈ 0.0000006667
            expect(result).toBeCloseTo(0.0000006667, 10);
        });
    });

    describe('mathematical properties', () => {
        it('should be non-negative', () => {
            const yTrue = [[-100], [0], [100], [999]];

            const result = mse.impurity(yTrue);
            expect(result).toBeGreaterThanOrEqual(0);
        });

        it('should be scale-invariant relative to the mean', () => {
            const yTrue1 = [[1], [2], [3]];
            const yTrue2 = [[11], [12], [13]]; // Same spread, different mean

            const result1 = mse.impurity(yTrue1);
            const result2 = mse.impurity(yTrue2);

            expect(result1).toBeCloseTo(result2, 6);
        });
    });
});
