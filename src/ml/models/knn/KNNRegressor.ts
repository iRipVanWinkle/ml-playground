import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { KNNParams, PredictionMetadata } from '../../types';
import { assertModelTrained } from '../../utils';
import { avgPreds, weightedAvgPreds } from '../../aggregators';
import { BaseKNN } from './BaseKNN';

export class KNNRegressor extends BaseKNN {
    async train(X: Tensor2D, y: Tensor2D): Promise<KNNParams> {
        return this.storeTrainingData(X, y);
    }

    predict(X: Tensor2D, params?: KNNParams): Tensor2D {
        const resolvedParams = params ?? this.params;
        assertModelTrained(resolvedParams);

        const predictions = tidy(() => {
            const { neighborValues, neighborDistanceWeights } = this.getNeighborBatch(
                X,
                resolvedParams,
            );

            if (this.weights === 'uniform') {
                return avgPreds(neighborValues);
            }

            return weightedAvgPreds(neighborValues, neighborDistanceWeights);
        });

        return predictions;
    }

    predictWithMetadata(X: Tensor2D, params?: KNNParams): PredictionMetadata {
        const result = this.predict(X, params);

        return {
            type: 'regression',
            predictions: result,
            dispose() {
                result.dispose();
            },
        };
    }
}
