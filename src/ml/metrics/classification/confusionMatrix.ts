import { oneHot, tidy, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Computes the confusion matrix for classification problems using only TensorFlow.js operations.
 * Expects one-hot encoded true labels and soft predictions (probabilities).
 *
 * @param yTrue - The true labels (shape: [n_samples, 1] for class indices)
 * @param yPred - The predicted labels (shape: [n_samples, 1] for class indices)
 * @returns The confusion matrix as a 2D tensor (shape: [numClasses, numClasses])
 */
export function confusionMatrix(yTrue: Tensor2D, yPred: Tensor2D, numClasses: number): Tensor2D {
    return tidy(() => {
        const yTrueFlat = yTrue.reshape([-1]).cast('int32');
        const yPredFlat = yPred.reshape([-1]).cast('int32');

        const yTrueHard = oneHot(yTrueFlat, numClasses);
        const yPredHard = oneHot(yPredFlat, numClasses);

        const cm = yTrueHard.transpose().matMul(yPredHard);
        return cm as Tensor2D;
    });
}
