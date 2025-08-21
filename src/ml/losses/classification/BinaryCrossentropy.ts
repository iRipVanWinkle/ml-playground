import { type Scalar, type Tensor2D, concat, log, neg, scalar, tidy } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { EPSILON } from '../../constants';

export class BinaryCrossentropy implements LossFunction {
    /**
     * Computes the binary cross-entropy (logistic) loss for binary classification.
     *
     * The loss is computed as:
     *   L(y_true, y_pred) = - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
     *
     * where:
     *   - y_true: true binary labels (0 or 1) (shape: [n_samples, 1])
     *   - y_pred: predicted probabilities (between 0 and 1) (shape: [n_samples, 1])
     *
     * The loss is averaged over all samples.
     *
     * @param yTrue - The true binary labels (shape: [n_samples, 1]).
     * @param yPred - The predicted probabilities (between 0 and 1).
     * @returns Scalar representing the binary cross-entropy loss.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => {
            const one = scalar(1);
            const yPredClipped = yPred.clipByValue(EPSILON, 1 - EPSILON);

            return neg(
                yTrue.mul(log(yPredClipped)).add(one.sub(yTrue).mul(log(one.sub(yPredClipped)))),
            ).mean();
        });
    }

    /**
     * Computes the gradient of the logistic loss (binary cross-entropy loss) function with respect to the model parameters.
     *
     * The gradients are computed as follows:
     *   - For the bias term:
     *       ∇BCE_bias = (1/n) * Σ (y_pred - y_true)
     *   - For the weights:
     *       ∇BCE_weights = (1/n) * Σ (x * (y_pred - y_true))
     *
     * where:
     *   - n: number of samples
     *   - x: feature matrix (shape: [n_samples, n_features])
     *   - y_true: true binary labels (0 or 1) (shape: [n_samples, 1])
     *   - y_pred: predicted probabilities (between 0 and 1) (shape: [n_samples, 1])
     *
     * @param xTrue - The feature matrix (shape: [n_samples, n_features]).
     * @param yTrue - The true binary labels (shape: [n_samples, 1]).
     * @param yPred - The predicted probabilities (between 0 and 1).
     * @returns Tensor2D containing the gradients.
     */
    parameterGradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        const sampleCount = xTrue.shape[0];

        return tidy(() => {
            // Calculate the gradient of the logistic loss
            const errors = yPred.sub(yTrue);

            // Compute the bias gradient
            const biasGrad = errors.sum().div(sampleCount);

            // Compute the weight gradient
            const weightGrad = xTrue.transpose().matMul(errors).div(sampleCount); // shape: [n_features, 1]

            // Concatenate bias gradient + feature gradients into one vector
            const gradients = concat([biasGrad.reshape([1, 1]), weightGrad]);

            return gradients as Tensor2D;
        });
    }

    /**
     * Computes the gradient of the binary cross-entropy (logistic) loss function with respect to the predictions.
     *
     * The gradient is computed as:
     *   grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
     *
     * where:
     *   - y_true: true binary labels (0 or 1) (shape: [n_samples, 1])
     *   - y_pred: predicted probabilities (between 0 and 1) (shape: [n_samples, 1])
     *
     * @param yTrue - The true binary labels (shape: [n_samples, 1]).
     * @param yPred - The predicted probabilities (between 0 and 1).
     * @returns Tensor2D containing the gradients of the binary cross-entropy loss with respect to the predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return tidy(() => {
            const yPredClipped = yPred.clipByValue(EPSILON, 1 - EPSILON);

            return yPredClipped.sub(yTrue).div(yPredClipped.mul(scalar(1).sub(yPredClipped)));
        });
    }
}
