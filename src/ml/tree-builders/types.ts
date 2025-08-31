import type { CriterionFunction, TrainingEventEmitter } from '../types';

export interface FeatureSelector {
    selectFeatures(features: number[][], seed: number): number[];
}

export interface SplitStrategy {
    findBestSplit(features: number[][], targets: number[][], indices: number[]): SplitResult | null;
}

export interface SplitResult {
    readonly featureIndex: number;
    readonly threshold: number;
    readonly leftIndices: number[];
    readonly rightIndices: number[];
    readonly impurityReduction: number;
}

export type CalculateNodeValueFn = (targets: number[][]) => {
    value: number;
    probabilities?: number[];
};

export type TreeBuilderOptions = {
    calculateNodeValueFn: CalculateNodeValueFn;
    splitStrategy: SplitStrategy;
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    eventEmitter?: TrainingEventEmitter;
};

export type SplitStrategyOptions = {
    featureSelector?: FeatureSelector;
    criterionFn: CriterionFunction;
    minSamplesLeaf: number;
};
