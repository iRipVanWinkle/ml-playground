import { describe, it, expect } from 'vitest';
import { recall, macroRecall, weightedRecall, binaryRecall } from './recall';

describe('recall', () => {
    describe('basic functionality', () => {
        it('should return 1.0 for perfect predictions', () => {
            const confusionMatrix: number[][] = [
                [3, 0, 0], // class 0: 3 correct
                [0, 2, 0], // class 1: 2 correct
                [0, 0, 1], // class 2: 1 correct
            ];

            const result = recall(confusionMatrix);
            expect(result).toEqual([1.0, 1.0, 1.0]);
        });

        it('should compute correct recall for each class', () => {
            // Class 0: 2 actual, 1 correct -> recall = 0.5
            // Class 1: 3 actual, 2 correct -> recall = 2/3
            // Class 2: 1 actual, 1 correct -> recall = 1.0
            const confusionMatrix: number[][] = [
                [1, 1, 0], // class 0: 1 correct, 1 misclassified as 1
                [1, 2, 0], // class 1: 2 correct, 1 misclassified as 0
                [0, 0, 1], // class 2: 1 correct
            ];

            const result = recall(confusionMatrix);
            expect(result[0]).toBeCloseTo(0.5); // 1/2
            expect(result[1]).toBeCloseTo(2 / 3); // 2/3
            expect(result[2]).toBe(1.0); // 1/1
        });

        it('should return 0 for classes with no actual instances', () => {
            const confusionMatrix: number[][] = [
                [2, 0, 0],
                [0, 0, 0], // class 1: no actual instances
                [0, 0, 0], // class 2: no actual instances
            ];

            const result = recall(confusionMatrix);
            expect(result).toEqual([1.0, 0, 0]);
        });
    });

    describe('binary classification', () => {
        it('should handle binary classification', () => {
            const confusionMatrix: number[][] = [
                [5, 1], // class 0: 5 correct, 1 misclassified -> recall = 5/6
                [2, 3], // class 1: 3 correct, 2 misclassified -> recall = 3/5
            ];

            const result = recall(confusionMatrix);
            expect(result[0]).toBeCloseTo(5 / 6);
            expect(result[1]).toBeCloseTo(3 / 5);
        });
    });
});

describe('macroRecall', () => {
    it('should compute macro average from confusion matrix', () => {
        const confusionMatrix: number[][] = [
            [2, 1, 0], // class 0: recall = 2/3
            [1, 2, 0], // class 1: recall = 2/3
            [0, 0, 1], // class 2: recall = 1/1
        ];

        const result = macroRecall(confusionMatrix);
        // (2/3 + 2/3 + 1) / 3
        expect(result).toBeCloseTo((2 / 3 + 2 / 3 + 1) / 3);
    });

    it('should compute macro average from recall array', () => {
        const recallArr = [0.5, 0.75, 1.0];
        const result = macroRecall(recallArr);
        expect(result).toBeCloseTo((0.5 + 0.75 + 1.0) / 3);
    });

    it('should return 0 for empty recall array', () => {
        const result = macroRecall([]);
        expect(result).toBe(0);
    });
});

describe('weightedRecall', () => {
    it('should compute weighted average from confusion matrix', () => {
        const confusionMatrix: number[][] = [
            [2, 1, 0], // class 0: recall = 2/3, actual = 3
            [1, 2, 0], // class 1: recall = 2/3, actual = 3
            [0, 0, 1], // class 2: recall = 1/1, actual = 1
        ];

        const result = weightedRecall(confusionMatrix);
        // Weighted: (2/3 * 3 + 2/3 * 3 + 1 * 1) / (3 + 3 + 1) = (2 + 2 + 1) / 7
        expect(result).toBeCloseTo(5 / 7);
    });

    it('should compute weighted average from recall array and confusion matrix', () => {
        const recallArr = [0.5, 0.75, 1.0];
        const confusionMatrix: number[][] = [
            [2, 1, 0], // actual: 3
            [1, 2, 0], // actual: 3
            [0, 0, 1], // actual: 1
        ];

        const result = weightedRecall(recallArr, confusionMatrix);
        // Row sums: [3, 3, 1]
        // Weighted: (0.5 * 3 + 0.75 * 3 + 1.0 * 1) / (3 + 3 + 1) = (1.5 + 2.25 + 1) / 7
        expect(result).toBeCloseTo(4.75 / 7);
    });
});

describe('binaryRecall', () => {
    it('should return 1.0 for perfect recall (no false negatives)', () => {
        // Matrix format: [[TP, FN], [FP, TN]]
        // TP = 10, FN = 0 -> recall = 10/10 = 1.0
        const binaryMatrix: number[][] = [
            [10, 0], // TP = 10, FN = 0
            [2, 5], // FP = 2, TN = 5
        ];

        const result = binaryRecall(binaryMatrix);
        expect(result).toBe(1.0);
    });

    it('should compute correct recall for typical case', () => {
        // TP = 8, FN = 2 -> recall = 8/(8+2) = 0.8
        const binaryMatrix: number[][] = [
            [8, 2], // TP = 8, FN = 2
            [1, 4], // FP = 1, TN = 4
        ];

        const result = binaryRecall(binaryMatrix);
        expect(result).toBe(0.8);
    });

    it('should return 0 when there are no true positives', () => {
        // TP = 0, FN = 5 -> recall = 0/(0+5) = 0
        const binaryMatrix: number[][] = [
            [0, 5], // TP = 0, FN = 5
            [3, 2], // FP = 3, TN = 2
        ];

        const result = binaryRecall(binaryMatrix);
        expect(result).toBe(0);
    });

    it('should return 0 when TP + FN = 0 (no actual positives)', () => {
        // TP = 0, FN = 0 -> recall = 0 (handled by zero division check)
        const binaryMatrix: number[][] = [
            [0, 0], // TP = 0, FN = 0
            [5, 3], // FP = 5, TN = 3
        ];

        const result = binaryRecall(binaryMatrix);
        expect(result).toBe(0);
    });

    it('should throw error for non-2x2 matrix', () => {
        const invalidMatrix1: number[][] = [
            [1, 2, 3],
            [4, 5, 6],
        ];
        const invalidMatrix2: number[][] = [[1], [2], [3]];
        const invalidMatrix3: number[][] = [[1, 2]];

        expect(() => binaryRecall(invalidMatrix1)).toThrow('Binary matrix must be 2x2');
        expect(() => binaryRecall(invalidMatrix2)).toThrow('Binary matrix must be 2x2');
        expect(() => binaryRecall(invalidMatrix3)).toThrow('Binary matrix must be 2x2');
    });
});
