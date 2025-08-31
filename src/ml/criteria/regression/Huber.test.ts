import { describe, it, expect, beforeEach } from 'vitest';
import { Huber } from './Huber';
import { MeanSquaredError } from './MeanSquaredError';

describe('Huber', () => {
    let huber: Huber;

    beforeEach(() => {
        huber = new Huber(1.0); // Default delta = 1.0
    });

    describe('impurity', () => {
        it('should return 0 for identical values', () => {
            const yTrue = [[5], [5], [5], [5]];

            const result = huber.impurity(yTrue);

            expect(result).toBeCloseTo(0, 6);
        });

        it('should behave like MSE for small errors (within delta)', () => {
            // Values within delta=1.0 from mean should use quadratic loss
            const yTrue = [[2], [2.5], [3], [3.5], [4]]; // mean = 3, all within ±1 of mean

            const result = huber.impurity(yTrue);

            // All errors are ≤ 1, so should be quadratic: 0.5 * error²
            // Errors from mean (3): [1, 0.5, 0, 0.5, 1]
            // Huber loss = 0.5 * (1² + 0.5² + 0² + 0.5² + 1²) / 5 = 0.5 * 2.5 / 5 = 0.25
            expect(result).toBeCloseTo(0.25, 6);
        });

        it('should behave like MAE for large errors (beyond delta)', () => {
            // Values far from mean should use linear loss
            const yTrue = [[0], [5]]; // mean = 2.5, errors = [2.5, 2.5], both > delta=1

            const result = huber.impurity(yTrue);

            // Both errors > delta, so linear: delta * |error| - 0.5 * delta²
            // For each: 1 * 2.5 - 0.5 * 1 = 2.5 - 0.5 = 2.0
            // Average: 2.0
            expect(result).toBeCloseTo(2.0, 6);
        });

        it('should handle mixed small and large errors', () => {
            // Mix of values within and beyond delta
            const yTrue = [[0], [2], [3], [4], [10]]; // mean = 3.8

            const result = huber.impurity(yTrue);

            // This tests the hybrid behavior of Huber loss
            expect(result).toBeGreaterThan(0);
            expect(Number.isFinite(result)).toBe(true);
        });

        it('should handle single value', () => {
            const yTrue = [[42]];

            const result = huber.impurity(yTrue);

            expect(result).toBeCloseTo(0, 6);
        });

        it('should be robust to outliers compared to MSE', () => {
            const yTrueWithOutlier = [[1], [2], [3], [100]];

            const huberResult = huber.impurity(yTrueWithOutlier);

            // Create MSE for comparison
            const mse = new MeanSquaredError();
            const mseResult = mse.impurity(yTrueWithOutlier);

            // Huber should be more robust (lower) than MSE for outlier case
            expect(huberResult).toBeLessThan(mseResult);

            mse.dispose();
        });

        it('should handle negative values', () => {
            const yTrue = [[-5], [-1], [0], [1], [5]];

            const result = huber.impurity(yTrue);

            expect(result).toBeGreaterThan(0);
            expect(Number.isFinite(result)).toBe(true);
        });

        it('should respect different delta values', () => {
            const yTrue = [[0], [5]]; // mean = 2.5, errors = [2.5, 2.5]

            // Test with different delta values
            const huber1 = new Huber(1.0);
            const huber2 = new Huber(3.0);

            const result1 = huber1.impurity(yTrue);
            const result2 = huber2.impurity(yTrue);

            // With delta=3.0, errors 2.5 are within delta, so should be quadratic (lower loss)
            // With delta=1.0, errors 2.5 are beyond delta, so should be linear (higher loss)
            expect(result2).toBeGreaterThan(result1);

            huber1.dispose();
            huber2.dispose();
        });
    });

    describe('mathematical properties', () => {
        it('should be non-negative', () => {
            const yTrue = [[-100], [0], [100], [999]];

            const result = huber.impurity(yTrue);
            expect(result).toBeGreaterThanOrEqual(0);
        });

        it('should be smooth at delta boundary', () => {
            // Test continuity at the delta boundary
            const yTrue = [[0], [1], [1], [1]]; // mean = 0.75

            const result = huber.impurity(yTrue);
            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThanOrEqual(0);
        });

        it('should converge to MSE as delta approaches infinity', () => {
            const yTrue = [[1], [2], [3], [4]];

            const huberLargeDelta = new Huber(1000);
            const mse = new MeanSquaredError();

            const huberResult = huberLargeDelta.impurity(yTrue);
            const mseResult = mse.impurity(yTrue);

            // With very large delta, Huber should approximate 0.5 * MSE closely
            expect(Math.abs(huberResult - 0.5 * mseResult)).toBeLessThan(0.001);

            huberLargeDelta.dispose();
            mse.dispose();
        });
    });
});
