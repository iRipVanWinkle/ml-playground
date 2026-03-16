import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import { EPSILON } from '../constants';

export function weightedAvgPreds(predictions: Tensor2D, weights: Tensor2D): Tensor2D {
    if (predictions.shape.length !== 2) {
        throw new Error('Predictions tensor must be 2D');
    }

    if (weights.shape.length !== 2) {
        throw new Error('Weights tensor must be 2D');
    }

    if (predictions.shape[0] !== weights.shape[0] || predictions.shape[1] !== weights.shape[1]) {
        throw new Error('Predictions and weights must have the same shape');
    }

    return tidy(() => {
        const weightedSum = predictions.mul(weights).sum(1, true);
        const totalWeight = weights.sum(1, true).maximum(EPSILON);

        return weightedSum.div(totalWeight) as Tensor2D;
    });
}
