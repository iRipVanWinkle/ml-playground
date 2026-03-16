import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { KNNParams, PredictionMetadata } from '../../types';
import { assertModelTrained } from '../../utils';
import { classProportions, weightedClassProportions } from '../../aggregators';
import { BaseKNN } from './BaseKNN';

export class KNNClassifier extends BaseKNN {
    async train(X: Tensor2D, y: Tensor2D): Promise<KNNParams> {
        const yData = await y.data();
        const classSet = new Set<number>(yData);

        return this.storeTrainingData(X, y, Array.from(classSet));
    }

    predict(X: Tensor2D, params?: KNNParams): Tensor2D {
        const resolvedParams = params ?? this.params;
        assertModelTrained(resolvedParams);
        const numClasses = resolvedParams.classes.length;

        const predictions = tidy(() => {
            const { neighborValues, neighborDistanceWeights } = this.getNeighborBatch(
                X,
                resolvedParams,
            );

            if (this.weights === 'uniform') {
                return classProportions(neighborValues, numClasses)
                    .argMax(1)
                    .expandDims(1) as Tensor2D;
            }

            return weightedClassProportions(neighborValues, neighborDistanceWeights, numClasses)
                .argMax(1)
                .expandDims(1) as Tensor2D;
        });

        return predictions;
    }

    predictWithMetadata(X: Tensor2D, params?: KNNParams): PredictionMetadata {
        const resolvedParams = params ?? this.params;
        assertModelTrained(resolvedParams);

        const [predictions, probabilities] = tidy(() => {
            const numClasses = resolvedParams.classes.length;
            const { neighborValues, neighborDistanceWeights } = this.getNeighborBatch(
                X,
                resolvedParams,
            );

            if (this.weights === 'uniform') {
                const probs = classProportions(neighborValues, numClasses);
                const preds = probs.argMax(1).expandDims(1) as Tensor2D;

                return [preds, probs];
            }

            const probs = weightedClassProportions(
                neighborValues,
                neighborDistanceWeights,
                numClasses,
            );
            const preds = probs.argMax(1).expandDims(1) as Tensor2D;

            return [preds, probs];
        }) as [Tensor2D, Tensor2D];

        return {
            type: 'classification',
            predictions,
            probabilities,
            dispose() {
                predictions.dispose();
                probabilities.dispose();
            },
        };
    }
}
