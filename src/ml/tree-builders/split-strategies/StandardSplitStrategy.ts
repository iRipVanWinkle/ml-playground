import { BaseSplitStrategy } from './BaseSplitStrategy';

/**
 * Standard CART-style split strategy
 */
export class StandardSplitStrategy extends BaseSplitStrategy {
    protected generateCandidateThresholds(featureValues: number[]): number[] {
        const candidateThresholds: number[] = [];

        // Get unique thresholds to try
        const uniqueValues = [...new Set(featureValues)].sort((a, b) => a - b);

        // If there are fewer than 2 unique values, we cannot split
        if (uniqueValues.length < 2) return [];

        for (let i = 0; i < uniqueValues.length - 1; i++) {
            candidateThresholds.push((uniqueValues[i] + uniqueValues[i + 1]) / 2);
        }

        return candidateThresholds;
    }
}
