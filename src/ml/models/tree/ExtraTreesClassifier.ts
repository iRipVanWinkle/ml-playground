import type { ExtraTreesOptions } from './ExtraTreesRegressor';
import { RandomForestClassifier } from './RandomForestClassifier';
import {
    AllFeatureSelector,
    computeClassProbabilities,
    RandomFeatureSelector,
    RandomThresholdSplitStrategy,
} from '../../tree-builders';
import type { Tensor2D } from '@tensorflow/tfjs';
import type { EnsembleTree } from '../../types';

export class ExtraTreesClassifier extends RandomForestClassifier {
    private numRandomThresholds: number;

    constructor(options: ExtraTreesOptions) {
        super(options);

        this.numRandomThresholds = options.numRandomThresholds ?? 1;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree> {
        const featureSelector = this.maxFeatures
            ? new RandomFeatureSelector(this.maxFeatures)
            : new AllFeatureSelector();

        const splitStrategy = new RandomThresholdSplitStrategy({
            featureSelector,
            criterionFn: this.criterion,
            minSamplesLeaf: this.minSamplesLeaf,
            numRandomThresholds: this.numRandomThresholds,
        });

        const options = {
            splitStrategy,
            calculateNodeValueFn: computeClassProbabilities,
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            minSamplesLeaf: this.minSamplesLeaf,
        };

        // Train the ensemble trees
        this.trees = await super.trainTreeEnsemble(X, y, options);

        return this.trees;
    }
}
