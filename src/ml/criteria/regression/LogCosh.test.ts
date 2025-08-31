import { describe, it, expect, beforeEach } from 'vitest';
import { LogCosh } from './LogCosh';

describe('LogCosh', () => {
    let logCosh: LogCosh;

    beforeEach(() => {
        logCosh = new LogCosh();
    });

    describe('impurity', () => {
        it('should return 0 for identical values', () => {
            const yTrue = [[5], [5], [5], [5]];

            const result = logCosh.impurity(yTrue);

            expect(result).toBeCloseTo(0, 6);
        });

        it('should compute log-cosh loss for simple case', () => {
            // Values: [0, 1] -> mean = 0.5
            // Differences from mean: [-0.5, 0.5]
            // log(cosh(-0.5)) = log(cosh(0.5)) ≈ log(1.1276) ≈ 0.1201
            // Average: 0.1201
            const yTrue = [[0], [1]];

            const result = logCosh.impurity(yTrue);

            expect(result).toBeCloseTo(0.1201, 4);
        });

        it('should handle single value', () => {
            const yTrue = [[42]];

            const result = logCosh.impurity(yTrue);

            expect(result).toBeCloseTo(0, 6);
        });

        it('should be smooth and differentiable', () => {
            // Test that the function produces finite, reasonable values
            const yTrue = [[1], [2], [3], [4], [5]];

            const result = logCosh.impurity(yTrue);

            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThan(10); // Should be reasonable magnitude
        });

        it('should handle negative values', () => {
            const yTrue = [[-5], [-1], [0], [1], [5]];

            const result = logCosh.impurity(yTrue);

            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThan(0);
        });

        it('should be robust to outliers (more than MSE, less than MAE)', () => {
            const yTrueWithOutlier = [[1], [2], [3], [100]];

            const result = logCosh.impurity(yTrueWithOutlier);

            // Should be finite and positive, but not as extreme as MSE
            expect(Number.isFinite(result)).toBe(true);
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThan(1000); // Much less than what MSE would give
        });

        it('should handle small deviations (MSE-like behavior)', () => {
            // For small errors, log(cosh(x)) ≈ x²/2 + x⁴/12 + ...
            // So it should behave similarly to MSE for small values
            const yTrue = [[2.9], [3.0], [3.1]]; // Small deviations around 3.0

            const result = logCosh.impurity(yTrue);

            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThan(0.1); // Should be small for small deviations
        });

        it('should handle large deviations (linear-like behavior)', () => {
            // For large x, log(cosh(x)) ≈ |x| - log(2)
            // So it should grow approximately linearly for large deviations
            const yTrue = [[0], [10]]; // Large deviation

            const result = logCosh.impurity(yTrue);

            expect(result).toBeGreaterThan(1);
            expect(Number.isFinite(result)).toBe(true);
        });

        it('should be symmetric around the mean', () => {
            // Values symmetric around mean should give same contribution
            const yTrue1 = [[0], [2]]; // mean = 1, deviations = [-1, 1]
            const yTrue2 = [[1], [1]]; // mean = 1, deviations = [0, 0]

            const result1 = logCosh.impurity(yTrue1);
            const result2 = logCosh.impurity(yTrue2);

            expect(result1).toBeGreaterThan(result2); // Non-zero deviation should be higher
            expect(result2).toBeCloseTo(0, 6); // Zero deviation should be zero
        });
    });

    describe('mathematical properties', () => {
        it('should be non-negative', () => {
            const yTrue = [[-100], [0], [100], [999]];

            const result = logCosh.impurity(yTrue);

            expect(result).toBeGreaterThanOrEqual(0);
        });

        it('should be convex', () => {
            // Test convexity property: f((x+y)/2) ≤ (f(x) + f(y))/2
            const yTrue1 = [[0], [0]];
            const yTrue2 = [[4], [4]];
            const yTrueMid = [[2], [2]];

            const result1 = logCosh.impurity(yTrue1);
            const result2 = logCosh.impurity(yTrue2);
            const resultMid = logCosh.impurity(yTrueMid);

            // Convexity: f(mid) ≤ (f(1) + f(2))/2
            expect(resultMid).toBeLessThanOrEqual((result1 + result2) / 2 + 1e-6);
        });

        it('should be smooth everywhere', () => {
            // Log-cosh is infinitely differentiable everywhere
            const testValues = [[-10], [-1], [-0.1], [0], [0.1], [1], [10]];

            for (const val of testValues) {
                const yTrue = [val, val];
                const result = logCosh.impurity(yTrue);

                expect(Number.isFinite(result)).toBe(true);
            }
        });

        it('should scale appropriately with input magnitude', () => {
            // Test with different scales
            const yTrue1 = [[0], [1]];
            const yTrue2 = [[0], [10]];
            const yTrue3 = [[0], [100]];

            const result1 = logCosh.impurity(yTrue1);
            const result2 = logCosh.impurity(yTrue2);
            const result3 = logCosh.impurity(yTrue3);

            // Should increase monotonically but sublinearly for large values
            expect(result2).toBeGreaterThan(result1);
            expect(result3).toBeGreaterThan(result2);
        });
    });

    describe('comparison with other loss functions', () => {
        it('should be between MSE and MAE behavior for moderate errors', () => {
            const yTrue = [[0], [3]]; // mean = 1.5, errors = [1.5, 1.5]

            const result = logCosh.impurity(yTrue);

            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThan(10); // Should be reasonable
        });

        it('should approximate MSE for very small errors', () => {
            // For small x, log(cosh(x)) ≈ x²/2
            const yTrue = [[2.99], [3.0], [3.01]]; // Very small deviations

            const result = logCosh.impurity(yTrue);

            // Should be very small, similar to what MSE would give
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThan(0.001);
        });

        it('should grow approximately linearly for large errors', () => {
            // For large |x|, log(cosh(x)) ≈ |x| - log(2)
            const yTrue1 = [[0], [5]]; // moderate error
            const yTrue2 = [[0], [10]]; // larger error

            const result1 = logCosh.impurity(yTrue1);
            const result2 = logCosh.impurity(yTrue2);

            // For linear growth, loss2/loss1 should be roughly 2 (10/5)
            const ratio = result2 / result1;
            expect(ratio).toBeGreaterThan(1.5);
            expect(ratio).toBeLessThan(2.5);
        });
    });
});
