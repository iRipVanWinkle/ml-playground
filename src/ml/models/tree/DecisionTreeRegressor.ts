import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import { BaseDecisionTree } from '../base/BaseDecisionTree';
import type { EnsembleTree, PredictionMetadata } from '../../types';
import { computeMeanValue, findLeafNode, StandardSplitStrategy } from '../../tree-builders';
import { assertModelTrained } from '../../utils';

export class DecisionTreeRegressor extends BaseDecisionTree {
    async train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree> {
        const [XArray, yArray] = await this.prepareTrainingData(X, y);

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

        const tree = await this.treeBuilder.buildTree(XArray, yArray, options);

        this.trees = [tree];

        return this.trees;
    }

    predict(X: Tensor2D, trees?: EnsembleTree): Tensor2D {
        const ensembleTrees = trees ?? this.trees;
        assertModelTrained(ensembleTrees);

        const [rootNode] = ensembleTrees; // Use the first tree

        const samplesArray = X.arraySync();

        const predictions = [];
        for (const sampleFeatures of samplesArray) {
            const leafNode = findLeafNode(sampleFeatures, rootNode);

            predictions.push([leafNode.value]);
        }

        return tensor2d(predictions);
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
