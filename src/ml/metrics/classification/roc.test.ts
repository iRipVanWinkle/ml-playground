import { describe, it, expect } from 'vitest';
import { rocCurve, multiclassRocCurve } from './roc';
import type { MatrixLike } from '../../utils/matrix';

describe('rocCurve', () => {
    describe('basic functionality', () => {
        it('should compute ROC curve for perfect classifier', () => {
            // Perfect classifier: all positives have higher probability than negatives
            const yTrue = [1, 1, 0, 0];
            const yProb = [0.9, 0.8, 0.3, 0.2];

            const result = rocCurve(yTrue, yProb);

            expect(result.fpr.length).toBeGreaterThan(0);
            expect(result.tpr.length).toBeGreaterThan(0);
            expect(result.thresholds.length).toBeGreaterThan(0);
            expect(result.fpr.length).toBe(result.tpr.length);
            expect(result.fpr.length).toBe(result.thresholds.length);

            // Should start at (0, 0)
            expect(result.fpr[0]).toBe(0);
            expect(result.tpr[0]).toBe(0);

            // Should end at (1, 1)
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
        });

        it('should compute ROC curve for worst classifier', () => {
            // Worst classifier: all negatives have higher probability than positives
            const yTrue = [1, 1, 0, 0];
            const yProb = [0.2, 0.3, 0.8, 0.9];

            const result = rocCurve(yTrue, yProb);

            // Should still form a valid ROC curve
            expect(result.fpr.length).toBeGreaterThan(0);
            expect(result.tpr.length).toBeGreaterThan(0);
            expect(result.fpr[0]).toBe(0);
            expect(result.tpr[0]).toBe(0);
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
        });

        it('should compute ROC curve for random classifier', () => {
            // Random classifier: mixed probabilities
            const yTrue = [1, 0, 1, 0, 1, 0];
            const yProb = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

            const result = rocCurve(yTrue, yProb);

            expect(result.fpr.length).toBeGreaterThan(0);
            expect(result.tpr.length).toBeGreaterThan(0);
            expect(result.fpr[0]).toBe(0);
            expect(result.tpr[0]).toBe(0);
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
        });

        it('should handle single positive and single negative', () => {
            const yTrue = [1, 0];
            const yProb = [0.8, 0.3];

            const result = rocCurve(yTrue, yProb);

            expect(result.fpr.length).toBeGreaterThanOrEqual(2);
            expect(result.tpr.length).toBeGreaterThanOrEqual(2);
            expect(result.fpr[0]).toBe(0);
            expect(result.tpr[0]).toBe(0);
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
        });
    });

    describe('edge cases', () => {
        it('should handle all positives', () => {
            const yTrue = [1, 1, 1];
            const yProb = [0.9, 0.8, 0.7];

            const result = rocCurve(yTrue, yProb);

            // TPR should always be 1 (all are positives)
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
            // FPR should go from 0 to 1
            expect(result.fpr[0]).toBe(0);
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
        });

        it('should handle all negatives', () => {
            const yTrue = [0, 0, 0];
            const yProb = [0.9, 0.8, 0.7];

            const result = rocCurve(yTrue, yProb);

            // TPR should go from 0 to 1
            expect(result.tpr[0]).toBe(0);
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
            // FPR should always be 1 (all are negatives)
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
        });

        it('should handle empty arrays', () => {
            const yTrue: number[] = [];
            const yProb: number[] = [];

            const result = rocCurve(yTrue, yProb);

            // Should still return valid structure
            expect(Array.from(result.fpr)).toEqual([0, 1]);
            expect(Array.from(result.tpr)).toEqual([0, 1]);
            expect(Array.from(result.thresholds)).toEqual([1, 0]);
        });

        it('should have thresholds in descending order', () => {
            const yTrue = [1, 0, 1, 0, 1, 0];
            const yProb = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4];

            const result = rocCurve(yTrue, yProb);

            // Thresholds should be in descending order (except possibly the last one)
            for (let i = 1; i < result.thresholds.length - 1; i++) {
                expect(result.thresholds[i]).toBeLessThanOrEqual(result.thresholds[i - 1]);
            }
        });
    });

    describe('optimal threshold calculations', () => {
        it('should find Youden optimal index', () => {
            // Create a scenario where optimal threshold is clear
            const yTrue = [1, 1, 1, 0, 0, 0];
            const yProb = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2];

            const result = rocCurve(yTrue, yProb);

            expect(result.youdenOptimalIndex).toEqual(3);
        });

        it('should find closest to corner index', () => {
            const yTrue = [1, 1, 1, 0, 0, 0];
            const yProb = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2];

            const result = rocCurve(yTrue, yProb);

            expect(result.closestToCornerIndex).toEqual(3);
        });
    });

    describe('TPR and FPR calculations', () => {
        it('should correctly calculate TPR and FPR for known data', () => {
            // Simple case: 2 positives, 2 negatives
            // Positives: [0.9, 0.7], Negatives: [0.8, 0.6]
            // Sorted by prob (desc): [0.9, 0.8, 0.7, 0.6]
            // Labels: [1, 0, 1, 0]
            const yTrue = [1, 0, 1, 0];
            const yProb = [0.9, 0.8, 0.7, 0.6];

            const result = rocCurve(yTrue, yProb);

            // At threshold > 0.9: TP=0, FP=0 -> TPR=0, FPR=0
            expect(result.tpr[0]).toBe(0);
            expect(result.fpr[0]).toBe(0);

            // At threshold 0.9: TP=1, FP=0 -> TPR=0.5, FPR=0
            // At threshold 0.8: TP=1, FP=1 -> TPR=0.5, FPR=0.5
            // At threshold 0.7: TP=2, FP=1 -> TPR=1, FPR=0.5
            // At threshold 0.6: TP=2, FP=2 -> TPR=1, FPR=1

            // Verify ending point
            expect(result.tpr[result.tpr.length - 1]).toBe(1);
            expect(result.fpr[result.fpr.length - 1]).toBe(1);
        });
    });
});

describe('multiclassRocCurve', () => {
    describe('basic functionality', () => {
        it('should compute ROC curves for multiclass problem', () => {
            // 3 classes, 6 samples
            const yTrue: MatrixLike = {
                array: new Float32Array([0, 1, 2, 0, 1, 2]),
                shape: [6, 1],
            };
            const yProb: MatrixLike = {
                array: new Float32Array([
                    0.7,
                    0.2,
                    0.1, // class 0
                    0.2,
                    0.7,
                    0.1, // class 1
                    0.1,
                    0.2,
                    0.7, // class 2
                    0.8,
                    0.1,
                    0.1, // class 0
                    0.1,
                    0.8,
                    0.1, // class 1
                    0.1,
                    0.1,
                    0.8, // class 2
                ]),
                shape: [6, 3],
            };

            const result = multiclassRocCurve(yTrue, yProb);

            expect(result.curves.length).toBe(3);
            expect(result.classIndices).toEqual([0, 1, 2]);

            const round = (x: number) => Number(x.toFixed(1));

            const roundedThresholds = result.curves.map((curve) =>
                Array.from(curve.thresholds).map(round),
            );

            expect(roundedThresholds[0]).toEqual([1, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1]);
            expect(roundedThresholds[1]).toEqual([1, 0.8, 0.7, 0.2, 0.2, 0.1, 0.1]);
            expect(roundedThresholds[2]).toEqual([1, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1]);
        });

        it('should handle binary classification as multiclass', () => {
            // 2 classes, 4 samples
            const yTrue: MatrixLike = {
                array: new Float32Array([0, 1, 0, 1]),
                shape: [4, 1],
            };
            const yProb: MatrixLike = {
                array: new Float32Array([
                    0.9,
                    0.1, // class 0
                    0.2,
                    0.8, // class 1
                    0.8,
                    0.2, // class 0
                    0.3,
                    0.7, // class 1
                ]),
                shape: [4, 2],
            };

            const result = multiclassRocCurve(yTrue, yProb);

            expect(result.curves.length).toBe(2);
            expect(result.classIndices).toEqual([0, 1]);
        });
    });

    describe('edge cases', () => {
        it('should handle single class per sample', () => {
            const yTrue: MatrixLike = {
                array: new Float32Array([0, 0, 0]),
                shape: [3, 1],
            };
            const yProb: MatrixLike = {
                array: new Float32Array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
                shape: [3, 2],
            };

            const result = multiclassRocCurve(yTrue, yProb);

            expect(result.curves.length).toBe(2);
            expect(result.classIndices).toEqual([0, 1]);
        });

        it('should handle empty input', () => {
            const yTrue: MatrixLike = {
                array: new Float32Array([]),
                shape: [0, 1],
            };
            const yProb: MatrixLike = {
                array: new Float32Array([]),
                shape: [0, 3],
            };

            const result = multiclassRocCurve(yTrue, yProb);

            expect(result.curves.length).toBe(3);
            expect(result.classIndices).toEqual([0, 1, 2]);
        });
    });
});
