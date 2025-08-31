import type { FeatureSelector } from '../types';

export class AllFeatureSelector implements FeatureSelector {
    selectFeatures(features: number[][]): number[] {
        if (features.length === 0) {
            return [];
        }
        return Array.from({ length: features[0].length }, (_, i) => i);
    }
}
