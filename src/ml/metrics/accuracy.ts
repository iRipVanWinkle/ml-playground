import { equal, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Computes the accuracy metric.
 *
 * @param yTrue - The true labels.
 * @param yPred - The predicted labels.
 * @returns The accuracy as a scalar.
 */
export function accuracy(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
    return tidy(() => {
        const correct = equal(yTrue, yPred);

        return correct.mean();
    });
}
