import { tensor2d, tidy, type Tensor2D } from '@tensorflow/tfjs';
import { BaseDecisionTree } from '../base/BaseDecisionTree';
import type { EnsembleTree, PredictionMetadata } from '../../types';
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
        const resolvedTrees = trees ?? this.trees;

        assertModelTrained(resolvedTrees);

        const [rootNode] = resolvedTrees; // Use the first tree

        const samplesArray = X.arraySync();

        const treeProbabilitiesArray: number[][] = [];
        for (const sampleFeatures of samplesArray) {
            const leafNode = findLeafNode(sampleFeatures, rootNode);

            treeProbabilitiesArray.push(leafNode.probabilities!);
        }

        return tidy(() => {
            const probabilities = tensor2d(treeProbabilitiesArray);

            return probabilityToClassIndex(probabilities);
        });
    }

    predictWithMetadata(X: Tensor2D, trees?: EnsembleTree): PredictionMetadata {
        const resolvedTrees = trees ?? this.trees;

        assertModelTrained(resolvedTrees);

        const [rootNode] = resolvedTrees; // Use the first tree

        const samplesArray = X.arraySync();

        const treeProbabilitiesArray: number[][] = [];
        for (const sampleFeatures of samplesArray) {
            const leafNode = findLeafNode(sampleFeatures, rootNode);

            treeProbabilitiesArray.push(leafNode.probabilities!);
        }

        const classProbabilities = tensor2d(treeProbabilitiesArray);
        const predictedClassIndices = probabilityToClassIndex(classProbabilities);

        return {
            type: 'classification',
            predictions: predictedClassIndices,
            probabilities: classProbabilities,
            dispose() {
                predictedClassIndices.dispose();
                classProbabilities.dispose();
            },
        };
    }
}
