import { Randomizer } from '../../random/Randomizer';
import type { FeatureSelector } from '../types';

export class RandomFeatureSelector implements FeatureSelector {
    private maxFeatures: number;

    constructor(maxFeatures: number) {
        this.maxFeatures = maxFeatures;
    }

    selectFeatures(features: number[][], seed: number): number[] {
        if (features.length === 0 || features[0].length === 0) {
            return [];
        }

        const numFeatures = features[0].length;

        if (!this.maxFeatures || this.maxFeatures <= 0) {
            return Array.from({ length: numFeatures }, (_, i) => i);
        }

        const actualMax = Math.min(this.maxFeatures, numFeatures);
        const indices = Randomizer.randomUniqueNumber([actualMax], 0, numFeatures, 'int32', seed);
        const indicesArr = indices.arraySync() as number[];

        indices.dispose();

        return indicesArr;
    }
}
