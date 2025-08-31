import { describe, it, expect, beforeEach } from 'vitest';
import { AllFeatureSelector } from './AllFeatureSelector';

describe('AllFeatureSelector', () => {
    let selector: AllFeatureSelector;

    beforeEach(() => {
        selector = new AllFeatureSelector();
    });

    describe('selectFeatures', () => {
        it('should return all feature indices for a dataset with multiple features', () => {
            const features = [
                [1, 2, 3],
                [4, 5, 6],
            ];
            const result = selector.selectFeatures(features);
            expect(result).toEqual([0, 1, 2]);
        });

        it('should return all feature indices for a dataset with single feature', () => {
            const features = [[1], [4]];
            const result = selector.selectFeatures(features);
            expect(result).toEqual([0]);
        });

        it('should return empty array for empty features', () => {
            const features: number[][] = [];
            const result = selector.selectFeatures(features);
            expect(result).toEqual([]);
        });

        it('should return empty array for features with no columns', () => {
            const features = [[], []];
            const result = selector.selectFeatures(features);
            expect(result).toEqual([]);
        });
    });
});
