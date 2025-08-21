import { type Scalar, type Tensor2D, tidy, concat } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { EPSILON } from '../../constants';

export class CategoricalCrossentropy implements LossFunction {
    /**
     * Computes the categorical cross-entropy loss for multiclass classification (softmax regression).
     *
     * The loss is computed as:
     *   L(y_true, y_pred) = -sum(y_true * log(y_pred))
     *
     * where:
     *   - y_true: one-hot encoded true labels (shape: [n_samples, n_classes])
     *   - y_pred: predicted probabilities from softmax (sum to 1 across classes)
     *
     * The loss is averaged over all samples.
     *
     * @param yTrueOneHot - The true one-hot encoded labels (shape: [n_samples, n_classes]).
     * @param yPred - The predicted class probabilities.
     * @returns Scalar representing the categorical cross-entropy loss.
     */
    compute(yTrueOneHot: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => {
            const yPredClipped = yPred.clipByValue(EPSILON, 1);

            // Compute -log(prob of true class) = -sum(yTrueOneHot * log(yPred))
            const losses = yTrueOneHot.mul(yPredClipped.log()).sum(1).neg(); // [batchSize]

            return losses.mean(); // Return as Scalar
        });
    }

    /**
     * Computes the gradient of the categorical cross-entropy loss with respect to model parameters.
     *
     * The gradients are computed as follows:
     *   - For the bias term:
     *       ∇CCE_bias = mean(y_pred - y_true)
     *   - For the weights:
     *       ∇CCE_weights = (X^T * (y_pred - y_true)) / n
     *
     * where:
     *   - n: number of samples
     *   - X: feature matrix (shape: [n_samples, n_features])
     *   - y_true: one-hot encoded true labels (shape: [n_samples, n_classes])
     *   - y_pred: predicted probabilities from softmax (shape: [n_samples, n_classes])
     *
     * @param xTrue - The feature matrix (shape: [n_samples, n_features]).
     * @param yTrueOneHot - The true one-hot encoded labels (shape: [n_samples, n_classes]).
     * @param yPred - The predicted probabilities from the softmax function.
     * @returns Tensor2D containing the gradients (shape: [n_features + 1, n_classes]).
     */
    parameterGradient(xTrue: Tensor2D, yTrueOneHot: Tensor2D, yPred: Tensor2D): Tensor2D {
        const sampleCount = xTrue.shape[0];

        return tidy(() => {
            // Gradient for softmax regression: (y_pred - y_true)
            const errors = yPred.sub(yTrueOneHot); // [n_samples, n_classes]

            // Compute the bias gradient: mean over all samples for each class
            const biasGrad = errors.mean(0).reshape([1, -1]); // [1, n_classes]
            // Compute the weight gradient: (X^T * errors) / n_samples
            const weightGrad = xTrue.transpose().matMul(errors).div(sampleCount); // [n_features, n_classes]
            // Concatenate bias gradient + weight gradients into one tensor
            const gradients = concat([biasGrad, weightGrad], 0); // [n_features + 1, n_classes]

            return gradients as Tensor2D;
        });
    }

    /**
     * Computes the gradient of the categorical cross-entropy loss with respect to the predictions.
     *
     * The gradient is computed as:
     *   grad = y_pred - y_true
     *
     * where:
     *   - y_true: one-hot encoded true labels (shape: [n_samples, n_classes])
     *   - y_pred: predicted probabilities from softmax (shape: [n_samples, n_classes])
     *
     * @param yTrue - The true one-hot encoded labels (shape: [n_samples, n_classes]).
     * @param yPred - The predicted probabilities from the softmax function.
     * @returns Tensor2D representing the gradient of the loss with respect to the predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return yPred.sub(yTrue);
    }
}
