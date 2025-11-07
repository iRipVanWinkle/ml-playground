import { describe, it, expect } from 'vitest';
import { f1Scores, macroAverageF1Score, weightedAverageF1Score, binaryF1Score } from './f1Scores';

describe('f1Scores', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const confusionMatrix: number[][] = [
                [3, 0, 0], // class 0: perfect
                [0, 2, 0], // class 1: perfect
                [0, 0, 1], // class 2: perfect
            ];

            const result = f1Scores(confusionMatrix);
            expect(result).toEqual([1.0, 1.0, 1.0]);
        });

        it('should compute correct F1 scores from confusion matrix', () => {
            // Class 0: precision = 2/3, recall = 2/3, F1 = 2/3
            // Class 1: precision = 2/3, recall = 2/3, F1 = 2/3
            // Class 2: precision = 1/1, recall = 1/1, F1 = 1.0
            const confusionMatrix: number[][] = [
                [2, 1, 0],
                [1, 2, 0],
                [0, 0, 1],
            ];

            const result = f1Scores(confusionMatrix);
            expect(result[0]).toBeCloseTo(2 / 3);
            expect(result[1]).toBeCloseTo(2 / 3);
            expect(result[2]).toBe(1.0);
        });

        it('should compute F1 scores from precision and recall arrays', () => {
            const precisionArr = [0.5, 0.75, 1.0];
            const recallArr = [0.5, 0.75, 1.0];

            const result = f1Scores(precisionArr, recallArr);
            // F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
            // F1 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
            // F1 = 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
            expect(result[0]).toBe(0.5);
            expect(result[1]).toBe(0.75);
            expect(result[2]).toBe(1.0);
        });

        it('should return 0 when precision and recall are both 0', () => {
            const precisionArr = [0, 0.5, 1.0];
            const recallArr = [0, 0.5, 1.0];

            const result = f1Scores(precisionArr, recallArr);
            expect(result[0]).toBe(0);
            expect(result[1]).toBe(0.5);
            expect(result[2]).toBe(1.0);
        });
    });

    describe('binary classification', () => {
        it('should handle binary classification', () => {
            const confusionMatrix: number[][] = [
                [5, 1], // class 0: precision = 5/7, recall = 5/6
                [2, 3], // class 1: precision = 3/4, recall = 3/5
            ];

            const result = f1Scores(confusionMatrix);
            // Class 0: F1 = 2 * (5/7 * 5/6) / (5/7 + 5/6)
            // Class 1: F1 = 2 * (3/4 * 3/5) / (3/4 + 3/5)
            expect(result[0]).toBeGreaterThan(0);
            expect(result[1]).toBeGreaterThan(0);
        });
    });
});

describe('macroAverageF1Score', () => {
    it('should compute macro average from confusion matrix', () => {
        const confusionMatrix: number[][] = [
            [2, 1, 0], // class 0: F1 = 2/3
            [1, 2, 0], // class 1: F1 = 2/3
            [0, 0, 1], // class 2: F1 = 1.0
        ];

        const result = macroAverageF1Score(confusionMatrix);
        // (2/3 + 2/3 + 1) / 3
        expect(result).toBeCloseTo((2 / 3 + 2 / 3 + 1) / 3);
    });

    it('should compute macro average from precision and recall arrays', () => {
        const precisionArr = [0.5, 0.75, 1.0];
        const recallArr = [0.5, 0.75, 1.0];

        const result = macroAverageF1Score(precisionArr, recallArr);
        // F1 scores: [0.5, 0.75, 1.0]
        // Macro: (0.5 + 0.75 + 1.0) / 3
        expect(result).toBeCloseTo((0.5 + 0.75 + 1.0) / 3);
    });
});

describe('weightedAverageF1Score', () => {
    it('should compute weighted average from confusion matrix', () => {
        const confusionMatrix: number[][] = [
            [2, 1, 0], // class 0: F1 = 2/3, actual = 3
            [1, 2, 0], // class 1: F1 = 2/3, actual = 3
            [0, 0, 1], // class 2: F1 = 1.0, actual = 1
        ];

        const result = weightedAverageF1Score(confusionMatrix);
        // Weighted: (2/3 * 3 + 2/3 * 3 + 1 * 1) / (3 + 3 + 1) = (2 + 2 + 1) / 7
        expect(result).toBeCloseTo(5 / 7);
    });

    it('should compute weighted average from precision, recall arrays and confusion matrix', () => {
        const precisionArr = [0.5, 0.75, 1.0];
        const recallArr = [0.5, 0.75, 1.0];
        const confusionMatrix: number[][] = [
            [2, 1, 0], // actual: 3
            [1, 2, 0], // actual: 3
            [0, 0, 1], // actual: 1
        ];

        const result = weightedAverageF1Score(precisionArr, recallArr, confusionMatrix);
        // F1 scores: [0.5, 0.75, 1.0]
        // Row sums: [3, 3, 1]
        // Weighted: (0.5 * 3 + 0.75 * 3 + 1.0 * 1) / (3 + 3 + 1) = (1.5 + 2.25 + 1) / 7
        expect(result).toBeCloseTo(4.75 / 7);
    });
});

describe('binaryF1Score', () => {
    describe('from binary matrix', () => {
        it('should return 1.0 for perfect F1 (perfect precision and recall)', () => {
            // Matrix format: [[TP, FN], [FP, TN]]
            // TP = 10, FP = 0, FN = 0 -> precision = 1.0, recall = 1.0, F1 = 1.0
            const binaryMatrix: number[][] = [
                [10, 0], // TP = 10, FN = 0
                [0, 5], // FP = 0, TN = 5
            ];

            const result = binaryF1Score(binaryMatrix);
            expect(result).toBe(1.0);
        });

        it('should compute correct F1 score from binary matrix', () => {
            // TP = 8, FP = 2, FN = 2
            // precision = 8/(8+2) = 0.8, recall = 8/(8+2) = 0.8
            // F1 = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8
            const binaryMatrix: number[][] = [
                [8, 2], // TP = 8, FN = 2
                [2, 4], // FP = 2, TN = 4
            ];

            const result = binaryF1Score(binaryMatrix);
            expect(result).toBeCloseTo(0.8);
        });

        it('should return 0 when precision and recall are both 0', () => {
            // TP = 0, FP = 5, FN = 5
            // precision = 0, recall = 0, F1 = 0
            const binaryMatrix: number[][] = [
                [0, 5], // TP = 0, FN = 5
                [5, 0], // FP = 5, TN = 0
            ];

            const result = binaryF1Score(binaryMatrix);
            expect(result).toBe(0);
        });

        it('should handle case where precision != recall', () => {
            // TP = 5, FP = 5, FN = 0
            // precision = 5/(5+5) = 0.5, recall = 5/(5+0) = 1.0
            // F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
            const binaryMatrix: number[][] = [
                [5, 0], // TP = 5, FN = 0
                [5, 0], // FP = 5, TN = 0
            ];

            const result = binaryF1Score(binaryMatrix);
            expect(result).toBeCloseTo(2 / 3);
        });

        it('should throw error for non-2x2 matrix', () => {
            const invalidMatrix1: number[][] = [
                [1, 2, 3],
                [4, 5, 6],
            ];
            const invalidMatrix2: number[][] = [[1], [2], [3]];
            const invalidMatrix3: number[][] = [[1, 2]];

            expect(() => binaryF1Score(invalidMatrix1)).toThrow('Binary matrix must be 2x2');
            expect(() => binaryF1Score(invalidMatrix2)).toThrow('Binary matrix must be 2x2');
            expect(() => binaryF1Score(invalidMatrix3)).toThrow('Binary matrix must be 2x2');
        });
    });

    describe('from precision and recall values', () => {
        it('should return 1.0 for perfect precision and recall', () => {
            const result = binaryF1Score(1.0, 1.0);
            expect(result).toBe(1.0);
        });

        it('should compute correct F1 score from precision and recall', () => {
            // F1 = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8
            const result = binaryF1Score(0.8, 0.8);
            expect(result).toBeCloseTo(0.8);
        });

        it('should handle different precision and recall values', () => {
            // F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
            const result = binaryF1Score(0.5, 1.0);
            expect(result).toBeCloseTo(2 / 3);
        });

        it('should handle case where only precision is 0', () => {
            // F1 = 2 * (0 * 0.8) / (0 + 0.8) = 0
            const result = binaryF1Score(0, 0.8);
            expect(result).toBe(0);
        });
    });
});
