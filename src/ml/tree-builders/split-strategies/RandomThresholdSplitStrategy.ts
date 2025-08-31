import { Randomizer } from '../../random/Randomizer';
import { BaseSplitStrategy } from './BaseSplitStrategy';
import type { SplitStrategyOptions } from '../types';
import { assert } from '../../utils';

type RandomThresholdSplitStrategyOptions = SplitStrategyOptions & { numRandomThresholds: number };

/**
 * Random threshold split strategy (for Extra Trees)
 */
export class RandomThresholdSplitStrategy extends BaseSplitStrategy {
    private numRandomThresholds: number;

    constructor(options: RandomThresholdSplitStrategyOptions) {
        super(options);

        assert(
            options.numRandomThresholds > 0,
            'numRandomThresholds must be greater than 0 for RandomThresholdSplitStrategy',
        );

        this.numRandomThresholds = options.numRandomThresholds;
    }

    protected generateCandidateThresholds(featureValues: number[], seed: number): number[] {
        const candidateThresholds: number[] = [];
        const minValue = Math.min(...featureValues);
        const maxValue = Math.max(...featureValues);

        // No variability in this feature
        if (minValue === maxValue) return [];

        const randomThresholds = Randomizer.randomUniform(
            [this.numRandomThresholds],
            minValue,
            maxValue,
            'float32',
            seed,
        );

        candidateThresholds.push(...(randomThresholds.arraySync() as number[]));

        randomThresholds.dispose();

        return candidateThresholds;
    }
}
