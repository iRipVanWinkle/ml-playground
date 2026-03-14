import { describe, it, expect, beforeEach } from 'vitest';
import { RandomThresholdSplitStrategy } from './RandomThresholdSplitStrategy';
import { Gini } from '../../criteria/classification/Gini';

describe('RandomThresholdSplitStrategy', () => {
    let strategy: RandomThresholdSplitStrategy;
    let criterion: Gini;

    beforeEach(() => {
        criterion = new Gini();
        strategy = new RandomThresholdSplitStrategy({
            criterionFn: criterion,
            minSamplesLeaf: 1,
            numRandomThresholds: 5,
        });
    });

    describe('constructor', () => {
        it('should create instance with valid numRandomThresholds', () => {
            const strat = new RandomThresholdSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
                numRandomThresholds: 10,
            });
            expect(strat).toBeInstanceOf(RandomThresholdSplitStrategy);
        });

        it('should throw error for numRandomThresholds <= 0', () => {
            expect(() => {
                new RandomThresholdSplitStrategy({
                    criterionFn: criterion,
                    minSamplesLeaf: 1,
                    numRandomThresholds: 0,
                });
            }).toThrow('numRandomThresholds must be greater than 0');
        });

        it('should throw error for negative numRandomThresholds', () => {
            expect(() => {
                new RandomThresholdSplitStrategy({
                    criterionFn: criterion,
                    minSamplesLeaf: 1,
                    numRandomThresholds: -1,
                });
            }).toThrow('numRandomThresholds must be greater than 0');
        });
    });

    describe('findBestSplit', () => {
        it('should find best split using random thresholds', () => {
            const features = [[1], [2], [3], [4], [5]];
            const targets = [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3, 4];
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
            const strategyWithMinLeaf = new RandomThresholdSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 3,
                numRandomThresholds: 5,
            });
            const features = [[1], [2], [3], [4]];
            const targets = [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3];
            const result = strategyWithMinLeaf.findBestSplit(indices, features, targets);
            expect(result).toBeNull(); // Cannot split without violating minSamplesLeaf
        });

        it('should handle different numRandomThresholds values', () => {
            const strategyWithMoreThresholds = new RandomThresholdSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
                numRandomThresholds: 20,
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
            const result = strategyWithMoreThresholds.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
        });

        it('should handle single threshold', () => {
            const strategyWithOneThreshold = new RandomThresholdSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
                numRandomThresholds: 1,
            });
            const features = [[1], [2], [3], [4]];
            const targets = [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ];
            const indices = [0, 1, 2, 3];
            const result = strategyWithOneThreshold.findBestSplit(indices, features, targets);
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
    });
});
