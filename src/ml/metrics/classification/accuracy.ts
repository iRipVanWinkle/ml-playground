import { equal, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Computes the accuracy metric from true and predicted labels.
 *
 * @param yTrue - The true labels.
 * @param yPred - The predicted labels.
 * @returns The accuracy as a scalar.
 */
export function accuracy(yTrue: Tensor2D, yPred: Tensor2D): Scalar;

/**
 * Computes the accuracy metric from a confusion matrix.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The accuracy as a number.
 */
export function accuracy(confusionMatrix: number[][]): number;

/**
 * Computes the accuracy metric.
 *
 * @param yTrueOrConfusionMatrix - The true labels or a confusion matrix.
 * @param yPred - The predicted labels (optional, only used when first param is Tensor2D).
 * @returns The accuracy as a scalar or number.
 */
export function accuracy(
    yTrueOrConfusionMatrix: Tensor2D | number[][],
    yPred?: Tensor2D,
): Scalar | number {
    // If first argument is a confusion matrix (number[][])
    if (Array.isArray(yTrueOrConfusionMatrix) && Array.isArray(yTrueOrConfusionMatrix[0])) {
        const matrix = yTrueOrConfusionMatrix;
        const numClasses = matrix.length;
        let correct = 0;
        let total = 0;

        // i = row = expected/actual class
        // j = col = predicted class
        for (let i = 0; i < numClasses; i++) {
            for (let j = 0; j < numClasses; j++) {
                total += matrix[i][j];
                if (i === j) {
                    correct += matrix[i][j];
                }
            }
        }

        return total === 0 ? 0 : correct / total;
    }

    // Original implementation for Tensor2D inputs
    const yTrue = yTrueOrConfusionMatrix as Tensor2D;
    if (!yPred) {
        throw new Error('yPred is required when yTrue is a Tensor2D');
    }

    return tidy(() => {
        const correct = equal(yTrue, yPred);
        return correct.mean() as Scalar;
    });
}
