import type { RandomForestOptions } from './RandomForestRegressor';
import { BaggingClassifier } from './BaggingClassifier';
import {
    AllFeatureSelector,
    computeClassProbabilities,
    RandomFeatureSelector,
    StandardSplitStrategy,
} from '../../tree-builders';
import type { Tensor2D } from '@tensorflow/tfjs';
import type { EnsembleTree } from '../../types';

export class RandomForestClassifier extends BaggingClassifier {
    protected maxFeatures?: number;

    constructor(options: RandomForestOptions) {
        super(options);

        this.maxFeatures = options.maxFeatures;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree> {
        const featureSelector = this.maxFeatures
            ? new RandomFeatureSelector(this.maxFeatures)
            : new AllFeatureSelector();

        const splitStrategy = new StandardSplitStrategy({
            featureSelector,
            criterionFn: this.criterion,
            minSamplesLeaf: this.minSamplesLeaf,
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
