import { BaseSplitStrategy } from './BaseSplitStrategy';
import type { SplitStrategyOptions } from '../types';
import { assert } from '../../utils';

type HistogramSplitStrategyOptions = SplitStrategyOptions & { maxBins?: number };

/**
 * Histogram-based split strategy for efficient splitting on large datasets
 */
export class HistogramSplitStrategy extends BaseSplitStrategy {
    private maxBins: number;

    constructor(options: HistogramSplitStrategyOptions) {
        super(options);

        const { maxBins = 256 } = options;

        assert(maxBins > 0, 'maxBins must be greater than 0 for HistogramSplitStrategy');

        this.maxBins = maxBins;
    }

    protected generateCandidateThresholds(featureValues: number[]): number[] {
        const candidateThresholds: number[] = [];
        const minValue = Math.min(...featureValues);
        const maxValue = Math.max(...featureValues);

        // No variability in this feature
        if (minValue === maxValue) return [];

        const numBins = Math.min(this.maxBins, Math.ceil(Math.sqrt(featureValues.length)));
        const binWidth = (maxValue - minValue) / numBins;

        for (let i = 1; i < numBins; i++) {
            candidateThresholds.push(minValue + i * binWidth);
        }

        return candidateThresholds;
    }
}
