import { describe, it, expect } from 'vitest';
import { precision, macroPrecision, weightedPrecision, binaryPrecision } from './precision';

describe('precision', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const confusionMatrix: number[][] = [
                [3, 0, 0], // class 0: 3 correct
                [0, 2, 0], // class 1: 2 correct
                [0, 0, 1], // class 2: 1 correct
            ];

            const result = precision(confusionMatrix);

            expect(result).toEqual([1.0, 1.0, 1.0]);
        });

        it('should compute correct precision for each class', () => {
            // Class 0: 2 predicted, 1 correct -> precision = 0.5
            // Class 1: 3 predicted, 2 correct -> precision = 2/3
            // Class 2: 1 predicted, 1 correct -> precision = 1.0
            const confusionMatrix: number[][] = [
                [1, 1, 0], // class 0: 1 correct as 0, 1 misclassified as 1
                [1, 2, 0], // class 1: 2 correct as 1, 1 misclassified as 0
                [0, 0, 1], // class 2: 1 correct
            ];

            const result = precision(confusionMatrix);

            expect(result[0]).toBeCloseTo(0.5); // 1/2
            expect(result[1]).toBeCloseTo(2 / 3); // 2/3
            expect(result[2]).toBe(1.0); // 1/1
        });

        it('should return 0 for classes with no predictions', () => {
            const confusionMatrix: number[][] = [
                [2, 0, 0],
                [0, 0, 0], // class 1: no predictions
                [0, 0, 0], // class 2: no predictions
            ];

            const result = precision(confusionMatrix);
            expect(result).toEqual([1.0, 0, 0]);
        });
    });

    describe('binary classification', () => {
        it('should handle binary classification', () => {
            const confusionMatrix: number[][] = [
                [5, 1], // class 0: 5 correct, 1 misclassified as 1
                [2, 3], // class 1: 3 correct, 2 misclassified as 0
            ];

            const result = precision(confusionMatrix);
            // Class 0: 5 correct out of 7 predicted = 5/7
            // Class 1: 3 correct out of 4 predicted = 3/4
            expect(result[0]).toBeCloseTo(5 / 7);
            expect(result[1]).toBeCloseTo(3 / 4);
        });
    });
});

describe('macroPrecision', () => {
    it('should compute macro average from confusion matrix', () => {
        const confusionMatrix: number[][] = [
            [2, 1, 0], // class 0: precision = 2/3
            [1, 2, 0], // class 1: precision = 2/3
            [0, 0, 1], // class 2: precision = 1/1
        ];

        const result = macroPrecision(confusionMatrix);
        // (2/3 + 2/3 + 1) / 3 = (2/3 + 2/3 + 1) / 3
        expect(result).toBeCloseTo((2 / 3 + 2 / 3 + 1) / 3);
    });

    it('should compute macro average from precision array', () => {
        const precisionArr = [0.5, 0.75, 1.0];
        const result = macroPrecision(precisionArr);
        expect(result).toBeCloseTo((0.5 + 0.75 + 1.0) / 3);
    });

    it('should return 0 for empty precision array', () => {
        const result = macroPrecision([]);
        expect(result).toBe(0);
    });
});

describe('weightedPrecision', () => {
    it('should compute weighted average from confusion matrix', () => {
        const confusionMatrix: number[][] = [
            [2, 1, 0], // class 0: precision = 2/3, predicted = 3
            [1, 2, 0], // class 1: precision = 2/3, predicted = 3
            [0, 0, 1], // class 2: precision = 1/1, predicted = 1
        ];

        const result = weightedPrecision(confusionMatrix);
        // Weighted: (2/3 * 3 + 2/3 * 3 + 1 * 1) / (3 + 3 + 1) = (2 + 2 + 1) / 7
        expect(result).toBeCloseTo(5 / 7);
    });

    it('should compute weighted average from precision array and confusion matrix', () => {
        const precisionArr = [0.5, 0.75, 1.0];
        const confusionMatrix: number[][] = [
            [2, 1, 0], // predicted: 3, 1, 0
            [1, 2, 0], // predicted: 3, 3, 0
            [0, 0, 1], // predicted: 0, 0, 1
        ];

        const result = weightedPrecision(precisionArr, confusionMatrix);
        // Column sums: [3, 3, 1]
        // Weighted: (0.5 * 3 + 0.75 * 3 + 1.0 * 1) / (3 + 3 + 1) = (1.5 + 2.25 + 1) / 7
        expect(result).toBeCloseTo(4.75 / 7);
    });
});

describe('binaryPrecision', () => {
    it('should return 1.0 for perfect precision (no false positives)', () => {
        // Matrix format: [[TP, FN], [FP, TN]]
        // TP = 10, FP = 0 -> precision = 10/10 = 1.0
        const binaryMatrix: number[][] = [
            [10, 0], // TP = 10, FN = 0
            [0, 5], // FP = 0, TN = 5
        ];

        const result = binaryPrecision(binaryMatrix);
        expect(result).toBe(1.0);
    });

    it('should compute correct precision for typical case', () => {
        // TP = 8, FP = 2 -> precision = 8/(8+2) = 0.8
        const binaryMatrix: number[][] = [
            [8, 1], // TP = 8, FN = 1
            [2, 4], // FP = 2, TN = 4
        ];

        const result = binaryPrecision(binaryMatrix);
        expect(result).toBe(0.8);
    });

    it('should return 0 when there are no true positives', () => {
        // TP = 0, FP = 5 -> precision = 0/(0+5) = 0
        const binaryMatrix: number[][] = [
            [0, 3], // TP = 0, FN = 3
            [5, 2], // FP = 5, TN = 2
        ];

        const result = binaryPrecision(binaryMatrix);
        expect(result).toBe(0);
    });

    it('should return 0 when TP + FP = 0 (no predictions)', () => {
        // TP = 0, FP = 0 -> precision = 0 (handled by zero division check)
        const binaryMatrix: number[][] = [
            [0, 5], // TP = 0, FN = 5
            [0, 3], // FP = 0, TN = 3
        ];

        const result = binaryPrecision(binaryMatrix);
        expect(result).toBe(0);
    });

    it('should throw error for non-2x2 matrix', () => {
        const invalidMatrix1: number[][] = [
            [1, 2, 3],
            [4, 5, 6],
        ];
        const invalidMatrix2: number[][] = [[1], [2], [3]];
        const invalidMatrix3: number[][] = [[1, 2]];

        expect(() => binaryPrecision(invalidMatrix1)).toThrow('Binary matrix must be 2x2');
        expect(() => binaryPrecision(invalidMatrix2)).toThrow('Binary matrix must be 2x2');
        expect(() => binaryPrecision(invalidMatrix3)).toThrow('Binary matrix must be 2x2');
    });
});
