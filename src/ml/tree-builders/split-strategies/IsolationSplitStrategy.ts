import { Randomizer } from '../../random/Randomizer';
import { getColumnValues, splitIndices } from '../helpers';
import type { SplitResult, SplitStrategy } from '../types';

/**
 * Isolation split strategy for the Isolation Forest algorithm.
 *
 * Instead of optimising for impurity reduction, this strategy randomly
 * selects one feature and a uniformly-random threshold between the
 * minimum and maximum values of that feature.  This is the split rule
 * described in the original Isolation Forest paper
 * (Liu, Ting & Zhou, 2008).
 *
 * @param baseSeed - Optional base seed added to the per-call counter so
 *   that different trees in the same forest use different random splits.
 */
export class IsolationSplitStrategy implements SplitStrategy {
    private baseSeed: number;
    private callCount: number = 0;

    constructor(baseSeed = 0) {
        this.baseSeed = baseSeed;
    }

    findBestSplit(indices: number[], features: number[][]): SplitResult | null {
        if (features.length === 0 || features[0].length === 0 || indices.length < 2) {
            return null;
        }

        const numFeatures = features[0].length;

        // Derive a unique seed for this call so each node in the tree gets
        // independently randomised splits.
        const seed = this.baseSeed + this.callCount * 37;
        this.callCount++;

        // Randomly select one feature (standard Isolation Forest algorithm).
        const featureIdxTensor = Randomizer.randomUniform([1], 0, numFeatures, 'int32', seed);
        const featureIndex = Math.floor((featureIdxTensor.arraySync() as number[])[0]);
        featureIdxTensor.dispose();

        const featureValues = getColumnValues(features, indices, featureIndex);
        const min = Math.min(...featureValues);
        const max = Math.max(...featureValues);

        // No variability in the selected feature — cannot split.
        if (min === max) return null;

        // Pick a uniformly random threshold in [min, max).
        const thresholdTensor = Randomizer.randomUniform(
            [1],
            min,
            max,
            'float32',
            seed + featureIndex + 1,
        );
        const threshold = (thresholdTensor.arraySync() as number[])[0];
        thresholdTensor.dispose();

        const { leftIndices, rightIndices } = splitIndices(featureValues, indices, threshold);

        if (leftIndices.length === 0 || rightIndices.length === 0) return null;

        return {
            featureIndex,
            threshold,
            leftIndices,
            rightIndices,
            impurityReduction: 1, // Isolation splits do not use impurity.
        };
    }
}
