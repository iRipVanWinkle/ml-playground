import { describe, it, expect, beforeEach } from 'vitest';
import { HistogramSplitStrategy } from './HistogramSplitStrategy';
import { Gini } from '../../criteria/classification/Gini';

describe('HistogramSplitStrategy', () => {
    let strategy: HistogramSplitStrategy;
    let criterion: Gini;

    beforeEach(() => {
        criterion = new Gini();
        strategy = new HistogramSplitStrategy({
            criterionFn: criterion,
            minSamplesLeaf: 1,
            maxBins: 10,
        });
    });

    describe('constructor', () => {
        it('should create instance with valid maxBins', () => {
            const strat = new HistogramSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
                maxBins: 256,
            });
            expect(strat).toBeInstanceOf(HistogramSplitStrategy);
        });

        it('should use default maxBins when not specified', () => {
            const strat = new HistogramSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
            });
            expect(strat).toBeInstanceOf(HistogramSplitStrategy);
        });

        it('should throw error for maxBins <= 0', () => {
            expect(() => {
                new HistogramSplitStrategy({
                    criterionFn: criterion,
                    minSamplesLeaf: 1,
                    maxBins: 0,
                });
            }).toThrow('maxBins must be greater than 0');
        });
    });

    describe('findBestSplit', () => {
        it('should find best split using histogram thresholds', () => {
            const features = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
            const targets = [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
            expect(result!.featureIndex).toBe(0);
            expect(typeof result!.threshold).toBe('number');
            expect(result!.leftIndices.length).toBeGreaterThan(0);
            expect(result!.rightIndices.length).toBeGreaterThan(0);
        });

        it('should return null when no valid splits found', () => {
            const features = [[1], [1], [1]];
            const targets = [
                [1, 0],
                [0, 1],
                [1, 0],
            ];
            const indices = [0, 1, 2];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).toBeNull();
        });

        it('should respect minSamplesLeaf constraint', () => {
            const strategyWithMinLeaf = new HistogramSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 6,
                maxBins: 10,
            });
            const features = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
            const targets = [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            const result = strategyWithMinLeaf.findBestSplit(indices, features, targets);
            expect(result).toBeNull(); // Cannot split without violating minSamplesLeaf
        });

        it('should handle different maxBins values', () => {
            const strategyWithMoreBins = new HistogramSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
                maxBins: 5,
            });
            const features = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
            const targets = [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            const result = strategyWithMoreBins.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
        });

        it('should handle small datasets', () => {
            const features = [[1], [2]];
            const targets = [
                [1, 0],
                [0, 1],
            ];
            const indices = [0, 1];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
        });

        it('should handle large datasets with binning', () => {
            const features = Array.from({ length: 100 }, (_, i) => [i]);
            const targets = Array.from({ length: 100 }, (_, i) => (i < 50 ? [1, 0] : [0, 1]));
            const indices = Array.from({ length: 100 }, (_, i) => i);
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
        });

        it('should handle multiple features', () => {
            const features = [
                [1, 10],
                [2, 20],
                [3, 30],
                [4, 40],
            ];
            const targets = [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
            expect([0, 1]).toContain(result!.featureIndex);
        });

        it('should handle default maxBins', () => {
            const strategyWithDefault = new HistogramSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
            });
            const features = [[1], [2], [3], [4], [5]];
            const targets = [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3, 4];
            const result = strategyWithDefault.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
        });
    });
});
