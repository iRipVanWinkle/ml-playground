import { describe, it, expect, beforeEach } from 'vitest';
import { StandardSplitStrategy } from './StandardSplitStrategy';
import { Gini } from '../../criteria/classification/Gini';
import { AllFeatureSelector } from '../feature-selectors/AllFeatureSelector';

describe('StandardSplitStrategy', () => {
    let strategy: StandardSplitStrategy;
    let criterion: Gini;

    beforeEach(() => {
        criterion = new Gini();
        strategy = new StandardSplitStrategy({
            criterionFn: criterion,
            minSamplesLeaf: 1,
        });
    });

    describe('constructor', () => {
        it('should create instance with default feature selector', () => {
            const strat = new StandardSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
            });
            expect(strat).toBeInstanceOf(StandardSplitStrategy);
        });

        it('should create instance with custom feature selector', () => {
            const featureSelector = new AllFeatureSelector();
            const strat = new StandardSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 1,
                featureSelector,
            });
            expect(strat).toBeInstanceOf(StandardSplitStrategy);
        });
    });

    describe('findBestSplit', () => {
        it('should return null for pure node', () => {
            const features = [[1], [2], [3]];
            const targets = [
                [1, 0],
                [1, 0],
                [1, 0],
            ]; // All same class
            const indices = [0, 1, 2];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).toBeNull();
        });

        it('should find best split for simple binary classification', () => {
            const features = [[1], [2], [3], [4]];
            const targets = [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ]; // Classes split at feature value 2.5
            const indices = [0, 1, 2, 3];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
            expect(result!.featureIndex).toBe(0);
            expect(result!.threshold).toBe(2.5);
            expect(result!.leftIndices).toEqual([0, 1]);
            expect(result!.rightIndices).toEqual([2, 3]);
        });

        it('should respect minSamplesLeaf constraint', () => {
            const strategyWithMinLeaf = new StandardSplitStrategy({
                criterionFn: criterion,
                minSamplesLeaf: 3,
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

        it('should return null when no valid splits found', () => {
            const features = [[1], [1], [1]]; // All same feature value
            const targets = [
                [1, 0],
                [0, 1],
                [1, 0],
            ];
            const indices = [0, 1, 2];
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).toBeNull();
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

        it('should handle partial indices', () => {
            const features = [[1], [2], [3], [4], [5], [6]];
            const targets = [
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ];
            const indices = [1, 2, 3, 4]; // Only middle 4 samples
            const result = strategy.findBestSplit(indices, features, targets);
            expect(result).not.toBeNull();
            expect(result!.featureIndex).toBe(0);
            expect(result!.threshold).toBe(3.5);
        });
    });
});
