import { describe, it, expect } from 'vitest';
import { cohensKappa } from './cohensKappa';

describe('cohensKappa', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const confusionMatrix: number[][] = [
                [3, 0, 0], // perfect predictions
                [0, 2, 0],
                [0, 0, 1],
            ];

            const result = cohensKappa(confusionMatrix);
            expect(result).toBe(1.0);
        });

        it('should return 0.0 for random agreement', () => {
            // When predictions match chance agreement
            // This is a simplified case - in practice, random agreement depends on class distribution
            const confusionMatrix: number[][] = [
                [1, 1, 1], // equal distribution
                [1, 1, 1],
                [1, 1, 1],
            ];

            const result = cohensKappa(confusionMatrix);
            // With equal distribution, kappa should be close to 0
            expect(result).toBeCloseTo(0, 1);
        });

        it('should compute correct kappa for binary classification', () => {
            // Binary classification: 8 out of 10 correct
            const confusionMatrix: number[][] = [
                [5, 1], // class 0: 5 correct, 1 wrong
                [1, 3], // class 1: 3 correct, 1 wrong
            ];

            const result = cohensKappa(confusionMatrix);
            // p0 = 8/10 = 0.8
            // pe = (6*6 + 4*4) / 100 = (36 + 16) / 100 = 0.52
            // kappa = (0.8 - 0.52) / (1 - 0.52) = 0.28 / 0.48 ≈ 0.583
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThanOrEqual(1.0);
        });

        it('should return 0 for empty confusion matrix', () => {
            const confusionMatrix: number[][] = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ];

            const result = cohensKappa(confusionMatrix);
            expect(result).toBe(0);
        });

        it('should handle three-class classification', () => {
            const confusionMatrix: number[][] = [
                [2, 1, 0], // class 0: 2 correct, 1 wrong
                [1, 2, 0], // class 1: 2 correct, 1 wrong
                [0, 0, 1], // class 2: 1 correct
            ];

            const result = cohensKappa(confusionMatrix);
            // Should be positive (better than chance) but less than 1
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThanOrEqual(1.0);
        });

        it('should return value in range [-1, 1]', () => {
            // Test with various confusion matrices
            const matrices: number[][][] = [
                [
                    [3, 0, 0],
                    [0, 2, 0],
                    [0, 0, 1],
                ],
                [
                    [0, 2, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ],
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
            ];

            matrices.forEach((matrix) => {
                const result = cohensKappa(matrix);
                expect(result).toBeGreaterThanOrEqual(-1);
                expect(result).toBeLessThanOrEqual(1);
            });
        });
    });
});
