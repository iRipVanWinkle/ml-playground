import type { CriterionFunction } from '../../../ml/types';
import { AllFeatureSelector } from '../feature-selectors';
import { gather, getColumnValues, splitIndices } from '../helpers';
import type { FeatureSelector, SplitResult, SplitStrategy, SplitStrategyOptions } from '../types';

export abstract class BaseSplitStrategy implements SplitStrategy {
    protected featureSelector: FeatureSelector;
    protected criterionFn: CriterionFunction;
    protected minSamplesLeaf: number;

    constructor(options: SplitStrategyOptions) {
        this.featureSelector = options.featureSelector ?? new AllFeatureSelector();
        this.criterionFn = options.criterionFn;
        this.minSamplesLeaf = options.minSamplesLeaf;
    }

    findBestSplit(
        features: number[][],
        targets: number[][],
        indices: number[],
    ): SplitResult | null {
        let bestSplit: SplitResult | null = null;
        let bestImpurityReduction = -Infinity;

        // Pre-compute parent impurity once
        const parentTargets = gather(targets, indices);
        const parentImpurity = this.computeImpurity(parentTargets);
        if (parentImpurity === 0) return null; // already pure

        const featureIndices = this.featureSelector.selectFeatures(features, parentImpurity);

        for (const featureIndex of featureIndices) {
            const split = this.findBestSplitForFeature(
                features,
                targets,
                indices,
                featureIndex,
                parentImpurity,
            );

            const isBestSplit =
                split &&
                (split.impurityReduction > bestImpurityReduction ||
                    (split.impurityReduction === bestImpurityReduction &&
                        split.featureIndex < bestSplit!.featureIndex)); // stable order

            if (isBestSplit) {
                bestImpurityReduction = split.impurityReduction;
                bestSplit = split;
            }
        }

        return bestSplit;
    }

    protected abstract generateCandidateThresholds(
        featureValues: number[],
        seed?: number,
    ): number[];

    private findBestSplitForFeature(
        features: number[][],
        targets: number[][],
        indices: number[],
        featureIndex: number,
        parentImpurity: number,
    ): SplitResult | null {
        // Get feature values for this feature and these indices
        const featureValues = getColumnValues(features, indices, featureIndex);
        const candidateThresholds = this.generateCandidateThresholds(featureValues, parentImpurity);

        // No thresholds to try
        if (candidateThresholds.length === 0) return null;

        let bestSplit: SplitResult | null = null;
        let bestImpurityReduction = -Infinity;

        // Try each threshold
        for (const threshold of candidateThresholds) {
            const { leftIndices, rightIndices } = splitIndices(featureValues, indices, threshold);

            if (
                leftIndices.length < this.minSamplesLeaf ||
                rightIndices.length < this.minSamplesLeaf
            ) {
                continue;
            }

            const impurityReduction = this.computeImpurityReduction(
                targets,
                indices,
                leftIndices,
                rightIndices,
                parentImpurity,
            );

            if (impurityReduction > bestImpurityReduction) {
                bestImpurityReduction = impurityReduction;
                bestSplit = {
                    featureIndex,
                    threshold,
                    leftIndices,
                    rightIndices,
                    impurityReduction,
                };
            }
        }

        return bestSplit;
    }

    private computeImpurityReduction(
        targets: number[][],
        parentIndices: number[],
        leftIndices: number[],
        rightIndices: number[],
        parentImpurity: number,
    ): number {
        const totalSamples = parentIndices.length;
        const leftSamples = leftIndices.length;
        const rightSamples = rightIndices.length;

        const leftTargets = gather(targets, leftIndices);
        const rightTargets = gather(targets, rightIndices);

        const leftImpurity = this.computeImpurity(leftTargets);
        const rightImpurity = this.computeImpurity(rightTargets);

        const leftWeight = leftSamples / totalSamples;
        const rightWeight = rightSamples / totalSamples;

        return parentImpurity - (leftImpurity * leftWeight + rightImpurity * rightWeight);
    }

    private computeImpurity(targets: number[][]): number {
        const impurity = this.criterionFn.impurity(targets);

        return impurity;
    }
}
