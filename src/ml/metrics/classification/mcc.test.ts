import { describe, it, expect } from 'vitest';
import { mcc } from './mcc';

describe('mcc', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const confusionMatrix: number[][] = [
                [3, 0, 0], // perfect predictions
                [0, 2, 0],
                [0, 0, 1],
            ];

            const result = mcc(confusionMatrix);
            expect(result).toBe(1.0);
        });

        it('should return 0 for empty confusion matrix', () => {
            const confusionMatrix: number[][] = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ];

            const result = mcc(confusionMatrix);
            expect(result).toBe(0);
        });

        it('should compute correct MCC for binary classification', () => {
            // Binary classification: 8 out of 10 correct
            const confusionMatrix: number[][] = [
                [5, 1], // class 0: 5 correct, 1 wrong
                [1, 3], // class 1: 3 correct, 1 wrong
            ];

            const result = mcc(confusionMatrix);
            // n = 10, sumDiagonal = 8
            // rowSums = [6, 4], colSums = [6, 4]
            // sumRowColProduct = 6*6 + 4*4 = 36 + 16 = 52
            // sumRowSumsSquared = 36 + 16 = 52
            // sumColSumsSquared = 36 + 16 = 52
            // numerator = 10*8 - 52 = 80 - 52 = 28
            // sqrtValue = (100 - 52) * (100 - 52) = 48 * 48 = 2304
            // denominator = sqrt(2304) = 48
            // MCC = 28/48 ≈ 0.5833
            expect(result).toBeCloseTo(28 / 48, 4);
        });
    });

    describe('binary classification', () => {
        it('should return 1.0 for perfect binary predictions', () => {
            const confusionMatrix: number[][] = [
                [5, 0], // class 0: 5 correct
                [0, 3], // class 1: 3 correct
            ];

            const result = mcc(confusionMatrix);
            expect(result).toBe(1.0);
        });

        it('should return -1.0 for worst possible binary predictions', () => {
            // All predictions are wrong (anti-diagonal)
            const confusionMatrix: number[][] = [
                [0, 5], // class 0: all misclassified as 1
                [3, 0], // class 1: all misclassified as 0
            ];

            const result = mcc(confusionMatrix);
            expect(result).toBe(-1.0);
        });

        it('should return 0 for random binary predictions', () => {
            // Equal distribution (random guessing)
            const confusionMatrix: number[][] = [
                [2, 2], // class 0: 2 correct, 2 wrong
                [2, 2], // class 1: 2 correct, 2 wrong
            ];

            const result = mcc(confusionMatrix);
            // With equal distribution, MCC should be close to 0
            expect(result).toBeCloseTo(0, 1);
        });

        it('should handle imbalanced binary classification', () => {
            const confusionMatrix: number[][] = [
                [10, 2], // class 0: 10 correct, 2 wrong
                [1, 5], // class 1: 5 correct, 1 wrong
            ];

            const result = mcc(confusionMatrix);
            // Should be positive (better than chance)
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThanOrEqual(1.0);
        });
    });

    describe('multiclass classification', () => {
        it('should handle four-class classification', () => {
            const confusionMatrix: number[][] = [
                [3, 0, 0, 0],
                [0, 2, 1, 0],
                [0, 0, 2, 1],
                [0, 0, 0, 1],
            ];

            const result = mcc(confusionMatrix);
            expect(result).toBeGreaterThan(0);
            expect(result).toBeLessThanOrEqual(1.0);
        });

        it('should handle case with some classes having no predictions', () => {
            const confusionMatrix: number[][] = [
                [5, 0, 0],
                [0, 0, 0], // class 1: no predictions
                [0, 0, 2], // class 2: 2 correct
            ];

            const result = mcc(confusionMatrix);
            // Should still compute a valid MCC
            expect(result).toBeGreaterThanOrEqual(-1);
            expect(result).toBeLessThanOrEqual(1);
        });
    });

    describe('edge cases', () => {
        it('should return 0 when sqrtValue is negative or zero', () => {
            // This tests the edge case where sqrtValue <= 0
            // This can happen with certain degenerate matrices
            const confusionMatrix: number[][] = [
                [1, 0],
                [0, 0], // Only one class has predictions
            ];

            const result = mcc(confusionMatrix);
            // Should handle gracefully and return 0
            expect(result).toBe(0);
        });

        it('should handle single class with all predictions', () => {
            const confusionMatrix: number[][] = [
                [10, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ];

            const result = mcc(confusionMatrix);
            // With only one class, MCC should be 0 (no meaningful correlation)
            expect(result).toBe(0);
        });
    });
});
