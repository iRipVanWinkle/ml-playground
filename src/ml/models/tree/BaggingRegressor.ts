import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type { EnsembleTree, PredictionMetadata } from '../../types';
import { averagePredictions } from '../../aggregators';
import { BaseEnsembleTree, type BaseEnsembleOptions } from '../base/BaseEnsembleTree';
import { computeMeanValue, findLeafNode, StandardSplitStrategy } from '../../tree-builders';
import { assertModelTrained } from '../../utils';

export class BaggingRegressor extends BaseEnsembleTree {
    constructor(options: BaseEnsembleOptions) {
        super(options);

        this.aggregator = options.aggregator ?? averagePredictions;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree> {
        const splitStrategy = new StandardSplitStrategy({
            criterionFn: this.criterion,
            minSamplesLeaf: this.minSamplesLeaf,
        });

        const options = {
            splitStrategy,
            calculateNodeValueFn: computeMeanValue,
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            minSamplesLeaf: this.minSamplesLeaf,
        };

        // Train the ensemble trees
        this.trees = await super.trainTreeEnsemble(X, y, options);

        return this.trees;
    }

    predict(X: Tensor2D, trees?: EnsembleTree): Tensor2D {
        assertModelTrained(trees ?? this.trees);

        const ensembleTrees = trees ?? this.trees!;

        const XArray = X.arraySync();

        const predictions = [];
        for (const sampleFeatures of XArray) {
            const rowPrediction: number[] = [];

            for (const rootNode of ensembleTrees) {
                const leafNode = findLeafNode(sampleFeatures, rootNode);

                rowPrediction.push(leafNode.value);
            }

            predictions.push(rowPrediction);
        }

        return this.aggregator(tensor2d(predictions));
    }

    predictWithMetadata(X: Tensor2D, trees?: EnsembleTree): PredictionMetadata {
        const prediction = this.predict(X, trees);

        return {
            type: 'regression',
            predictions: prediction,
            dispose() {
                prediction.dispose();
            },
        };
    }
}
