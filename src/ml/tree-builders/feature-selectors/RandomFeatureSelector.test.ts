import { describe, it, expect } from 'vitest';
import { RandomFeatureSelector } from './RandomFeatureSelector';

describe('RandomFeatureSelector', () => {
    let selector: RandomFeatureSelector;

    describe('constructor', () => {
        it('should create instance with maxFeatures', () => {
            selector = new RandomFeatureSelector(2);
            expect(selector).toBeInstanceOf(RandomFeatureSelector);
        });
    });

    describe('selectFeatures', () => {
        it('should return all feature indices when maxFeatures is 0', () => {
            selector = new RandomFeatureSelector(0);
            const features = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ];
            const result = selector.selectFeatures(features, 42);
            expect(result).toEqual([0, 1, 2, 3]);
        });

        it('should return all feature indices when maxFeatures is negative', () => {
            selector = new RandomFeatureSelector(-1);
            const features = [
                [1, 2, 3],
                [4, 5, 6],
            ];
            const result = selector.selectFeatures(features, 42);
            expect(result).toEqual([0, 1, 2]);
        });

        it('should return subset of features when maxFeatures is less than total', () => {
            selector = new RandomFeatureSelector(2);
            const features = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ];
            const result = selector.selectFeatures(features, 42);
            expect(result).toHaveLength(2);
            expect(result.every((index) => index >= 0 && index < 4)).toBe(true);
        });

        it('should return all features when maxFeatures exceeds total features', () => {
            selector = new RandomFeatureSelector(10);
            const features = [
                [1, 2, 3],
                [4, 5, 6],
            ];
            const result = selector.selectFeatures(features, 42).sort((a, b) => a - b);
            expect(result).toEqual([0, 1, 2]);
        });

        it('should be deterministic with the same seed', () => {
            selector = new RandomFeatureSelector(2);
            const features = [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ];
            const result1 = selector.selectFeatures(features, 123);
            const result2 = selector.selectFeatures(features, 123);
            expect(result1).toEqual(result2);
        });

        it('should return different results with different seeds', () => {
            selector = new RandomFeatureSelector(2);
            const features = [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ];
            const result1 = selector.selectFeatures(features, 123);
            const result2 = selector.selectFeatures(features, 456);
            // Note: This might occasionally fail if random selection happens to be the same, but it's rare
            expect(result1).not.toEqual(result2);
        });

        it('should return empty array for empty features', () => {
            selector = new RandomFeatureSelector(2);
            const features: number[][] = [];
            const result = selector.selectFeatures(features, 42);
            expect(result).toEqual([]);
        });

        it('should return empty array for features with no columns', () => {
            selector = new RandomFeatureSelector(2);
            const features = [[], []];
            const result = selector.selectFeatures(features, 42);
            expect(result).toEqual([]);
        });
    });
});
