import { type Scalar, type Tensor2D, concat, log, neg, scalar, tidy } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { EPSILON } from '../../constants';

export class BinaryCrossentropy implements LossFunction {
    /**
     * Logistic loss (also known as log loss or binary cross-entropy loss) is a loss function used in binary classification tasks.
     *
     * It measures the performance of a classification model whose output is a probability value between 0 and 1.
     *
     * Formula:
     *     L(y_true, y_pred) = - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
     *
     * where:
     *     - y_true: true binary labels (0 or 1)
     *     - y_pred: predicted probabilities (between 0 and 1)
     *
     * The loss is averaged over all samples.
     *
     * @param yTrue - The true binary labels (0 or 1).
     * @param yPred - The predicted probabilities (between 0 and 1).
     * @returns Scalar representing the logistic loss.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => {
            const one = scalar(1);
            const yPredClipped = yPred.clipByValue(EPSILON, 1 - EPSILON);

            return neg(
                yTrue.mul(log(yPredClipped)).add(one.sub(yTrue).mul(log(one.sub(yPredClipped)))),
            )
                .mean()
                .asScalar();
        });
    }

    /**
     * Computes the gradient of the logistic loss (binary cross-entropy loss) function with respect to the model parameters.
     *
     * The gradient is calculated as follows:
     *   - grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
     *
     * The gradients are computed as:
     *   - For the bias term:
     *       ∇L_bias = (1/n) * Σ [grad]
     *   - For the weights:
     *       ∇L_weights = (1/n) * Σ [x * grad]
     *
     * where:
     *   - n: number of samples
     *   - x: feature matrix
     *   - y_true: true binary labels (0 or 1)
     *   - y_pred: predicted probabilities (between 0 and 1)
     *
     * @param data - The DataModel containing features and labels.
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
     * Computes the gradient of the logistic loss (binary cross-entropy loss) function with respect to the predictions.
     *
     * The gradient is calculated as:
     *   - grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
     *
     * @param yTrue - The true binary labels (0 or 1).
     * @param yPred - The predicted probabilities (between 0 and 1).
     * @returns Tensor2D containing the gradients of the logistic loss with respect to the predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return tidy(() => {
            const yPredClipped = yPred.clipByValue(EPSILON, 1 - EPSILON);

            return yPredClipped.sub(yTrue).div(yPredClipped.mul(scalar(1).sub(yPredClipped)));
        });
    }
}
