import { tensor3d, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { EnsembleTree } from '../../types';
import { softVoting } from '../../aggregators';
import { BaseEnsembleTree, type BaseEnsembleOptions } from '../base/BaseEnsembleTree';
import {
    computeClassProbabilities,
    findLeafNode,
    probabilityToClassIndex,
    StandardSplitStrategy,
} from '../../tree-builders';
import { assertModelTrained } from '../../utils';

export class BaggingClassifier extends BaseEnsembleTree {
    constructor(options: BaseEnsembleOptions) {
        super(options);

        this.aggregator = options.aggregator ?? softVoting;
    }

    usesOneHotLabels(): boolean {
        return true;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree> {
        const splitStrategy = new StandardSplitStrategy({
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

    predict(X: Tensor2D, trees?: EnsembleTree): Tensor2D {
        assertModelTrained(trees ?? this.trees);

        const ensembleTrees = trees ?? this.trees!;

        const XArray = X.arraySync();

        const predictions: number[][][] = [];
        for (const sampleFeatures of XArray) {
            const rowPrediction: number[][] = [];
            for (const rootNode of ensembleTrees) {
                const leafNode = findLeafNode(sampleFeatures, rootNode);

                rowPrediction.push(leafNode.probabilities!);
            }

            predictions.push(rowPrediction);
        }

        return tidy(() => {
            const predictionsMatrix = tensor3d(predictions);
            const probabilities = this.aggregator(predictionsMatrix);

            return probabilityToClassIndex(probabilities);
        });
    }

    evaluate(X: Tensor2D, y: Tensor2D, trees?: EnsembleTree): [Tensor2D, Tensor2D, Scalar] {
        assertModelTrained(trees ?? this.trees);

        const ensembleTrees = trees ?? this.trees!; // Use the first tree

        const XArray = X.arraySync();

        const predictions: number[][][] = [];
        for (const sampleFeatures of XArray) {
            const rowPrediction: number[][] = [];
            for (const rootNode of ensembleTrees) {
                const leafNode = findLeafNode(sampleFeatures, rootNode);

                rowPrediction.push(leafNode.probabilities!);
            }

            predictions.push(rowPrediction);
        }

        const result = tidy(() => {
            const predictionsMatrix = tensor3d(predictions);
            const probabilities = this.aggregator(predictionsMatrix);

            const yPred = probabilityToClassIndex(probabilities);

            // Compute default loss using the loss function
            const loss = this.criterion.loss(y, yPred);

            return [yPred, probabilities, loss] as [Tensor2D, Tensor2D, Scalar];
        });

        return result;
    }
}
