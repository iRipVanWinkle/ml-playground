import { tensor3d, tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { EnsembleTree, PredictionMetadata } from '../../types';
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
        const ensembleTrees = trees ?? this.trees;

        assertModelTrained(ensembleTrees);

        const samplesArray = X.arraySync();

        const treeProbabilitiesArray = this.calculatePrediction(samplesArray, ensembleTrees);

        return tidy(() => {
            const predictionsMatrix = tensor3d(treeProbabilitiesArray);
            const classProbabilities = this.aggregator(predictionsMatrix);

            return probabilityToClassIndex(classProbabilities);
        });
    }

    predictWithMetadata(X: Tensor2D, trees?: EnsembleTree): PredictionMetadata {
        const ensembleTrees = trees ?? this.trees;

        assertModelTrained(ensembleTrees);

        const samplesArray = X.arraySync();

        const treeProbabilitiesArray = this.calculatePrediction(samplesArray, ensembleTrees);

        const [classProbabilities, predictedClassIndices] = tidy(() => {
            const predictionsMatrix = tensor3d(treeProbabilitiesArray);
            const classProbabilities = this.aggregator(predictionsMatrix);

            const predictedClassIndices = probabilityToClassIndex(classProbabilities);

            return [classProbabilities, predictedClassIndices];
        });

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

    protected calculatePrediction(XArray: number[][], ensembleTrees: EnsembleTree): number[][][] {
        const predictions: number[][][] = [];
        for (const sampleFeatures of XArray) {
            const rowPrediction: number[][] = [];
            for (const rootNode of ensembleTrees) {
                const leafNode = findLeafNode(sampleFeatures, rootNode);

                rowPrediction.push(leafNode.probabilities!);
            }

            predictions.push(rowPrediction);
        }

        return predictions;
    }
}
