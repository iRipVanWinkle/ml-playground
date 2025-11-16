import { describe, it, expect } from 'vitest';
import { auc } from './auc';

describe('auc', () => {
    describe('basic functionality', () => {
        it('should compute AUC for a simple ROC curve', () => {
            // Simple case: (0,0) -> (0.5, 0.5) -> (1, 1)
            // Area = 0.5 * 0.5 * 0.5 + 0.5 * (0.5 + 1) / 2 = 0.125 + 0.375 = 0.5
            const fpr = [0, 0.5, 1];
            const tpr = [0, 0.5, 1];

            const result = auc(fpr, tpr);
            expect(result).toBeCloseTo(0.5, 5);
        });

        it('should compute AUC for perfect classifier', () => {
            // Perfect classifier: goes from (0,0) to (0,1) to (1,1)
            // Area = 1.0 (entire area under the curve)
            const fpr = [0, 0, 1];
            const tpr = [0, 1, 1];

            const result = auc(fpr, tpr);
            expect(result).toBeCloseTo(1.0, 5);
        });

        it('should compute AUC for worst classifier', () => {
            // Worst classifier: goes from (0,0) to (1,0) to (1,1)
            // Area = 0.0 (no area under the curve)
            const fpr = [0, 1, 1];
            const tpr = [0, 0, 1];

            const result = auc(fpr, tpr);
            expect(result).toBeCloseTo(0.0, 5);
        });

        it('should compute AUC for random classifier', () => {
            // Random classifier: diagonal line
            // Area should be approximately 0.5
            const fpr = [0, 0.25, 0.5, 0.75, 1];
            const tpr = [0, 0.25, 0.5, 0.75, 1];

            const result = auc(fpr, tpr);
            expect(result).toBeCloseTo(0.5, 5);
        });

        it('should handle non-uniform FPR spacing', () => {
            // FPR: [0, 0.2, 0.8, 1]
            // TPR: [0, 0.6, 0.9, 1]
            // Area = 0.2 * (0 + 0.6) / 2 + 0.6 * (0.6 + 0.9) / 2 + 0.2 * (0.9 + 1) / 2
            //     = 0.2 * 0.3 + 0.6 * 0.75 + 0.2 * 0.95
            //     = 0.06 + 0.45 + 0.19 = 0.7
            const fpr = [0, 0.2, 0.8, 1];
            const tpr = [0, 0.6, 0.9, 1];

            const result = auc(fpr, tpr);
            expect(result).toBeCloseTo(0.7, 5);
        });

        it('should handle Float32Array inputs', () => {
            const fpr = Float32Array.from([0, 0.5, 1]);
            const tpr = Float32Array.from([0, 0.5, 1]);

            const result = auc(fpr, tpr);
            expect(result).toBeCloseTo(0.5, 5);
        });

        it('should return 0 for empty arrays', () => {
            const fpr: number[] = [];
            const tpr: number[] = [];

            const result = auc(fpr, tpr);
            expect(result).toBe(0);
        });
    });
});
