import { oneHot, tidy, type Tensor2D } from '@tensorflow/tfjs';
import { EPSILON } from '../constants';

export function weightedClassProportions(
    labels: Tensor2D,
    weights: Tensor2D,
    numClasses: number,
): Tensor2D {
    if (labels.shape.length !== 2) {
        throw new Error('Labels tensor must be 2D');
    }

    if (weights.shape.length !== 2) {
        throw new Error('Weights tensor must be 2D');
    }

    if (labels.shape[0] !== weights.shape[0] || labels.shape[1] !== weights.shape[1]) {
        throw new Error('Labels and weights must have the same shape');
    }

    if (!Number.isInteger(numClasses) || numClasses < 1) {
        throw new Error('numClasses must be a positive integer');
    }

    return tidy(() => {
        const [numSamples, numVotes] = labels.shape;

        const labelsFlat = labels.reshape([-1]).cast('int32');
        const hot = oneHot(labelsFlat, numClasses);

        const reshaped = hot.reshape([numSamples, numVotes, numClasses]);

        const weightsExpanded = weights.expandDims(2);

        const weightedSum = reshaped.mul(weightsExpanded).sum(1) as Tensor2D;

        const totalWeight = weights.sum(1, true).maximum(EPSILON);

        return weightedSum.div(totalWeight) as Tensor2D;
    });
}
