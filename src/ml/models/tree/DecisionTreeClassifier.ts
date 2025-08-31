import { tensor2d, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { BaseDecisionTree } from '../base/BaseDecisionTree';
import type { EnsembleTree } from '../../types';
import {
    computeClassProbabilities,
    findLeafNode,
    probabilityToClassIndex,
    StandardSplitStrategy,
} from '../../tree-builders';
import { assertModelTrained } from '../../utils';

export class DecisionTreeClassifier extends BaseDecisionTree {
    usesOneHotLabels(): boolean {
        return true;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree> {
        const [XArray, yArray] = await this.prepareTrainingData(X, y);

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

        const tree = await this.treeBuilder.buildTree(XArray, yArray, options);

        this.trees = [tree];

        return this.trees;
    }

    predict(X: Tensor2D, trees?: EnsembleTree): Tensor2D {
        assertModelTrained(trees ?? this.trees);

        const [rootNode] = trees ?? this.trees!; // Use the first tree

        const XArray = X.arraySync();

        const predictions: number[][] = [];
        for (const sampleFeatures of XArray) {
            const leafNode = findLeafNode(sampleFeatures, rootNode);

            predictions.push(leafNode.probabilities!);
        }

        return tidy(() => {
            const probabilities = tensor2d(predictions);

            return probabilityToClassIndex(probabilities);
        });
    }

    evaluate(X: Tensor2D, y: Tensor2D, trees?: EnsembleTree): [Tensor2D, Tensor2D, Scalar] {
        assertModelTrained(trees ?? this.trees);

        const [rootNode] = trees ?? this.trees!; // Use the first tree

        const XArray = X.arraySync();

        const predictions: number[][] = [];
        for (const sampleFeatures of XArray) {
            const leafNode = findLeafNode(sampleFeatures, rootNode);
            predictions.push(leafNode.probabilities!);
        }

        const result = tidy(() => {
            const probabilities = tensor2d(predictions);

            const yPred = probabilityToClassIndex(probabilities);

            // Compute default loss using the loss function
            const loss = this.criterion.loss(y, yPred);

            return [yPred, probabilities, loss] as [Tensor2D, Tensor2D, Scalar];
        });

        return result;
    }
}
