import { type Scalar, type Tensor2D, concat, tidy } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';

export class MeanSquaredError implements LossFunction {
    /**
     * Mean Squared Error (MSE) is a common loss function used in regression tasks.
     *
     * It measures the average of the squares of the errors, which is the difference
     * between the predicted and actual values.
     *
     * The formula for MSE is:
     *     MSE = (1/n) * Σ(y_true - y_pred)²
     *
     * where:
     *     - n is the number of samples
     *     - y_true is the true value
     *     - y_pred is the predicted value
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the Mean Squared Error.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => yPred.sub(yTrue).square().mean());
    }

    /**
     * Computes the gradient of the Mean Squared Error (MSE) loss function with respect to the parameters.
     *
     * The gradients are computed as follows:
     *   - For the bias term:
     *       ∇MSE_bias = (1/n) * Σ (y_pred - y_true)
     *   - For the weights:
     *       ∇MSE_weights = (1/n) * Σ [x * (y_pred - y_true)]
     *
     * where:
     *   - n: number of samples
     *   - x: feature matrix
     *   - y_true: true values (labels)
     *   - y_pred: predicted values
     *
     * @param xTrue - The feature matrix (shape: [n_samples, n_features]).
     * @param yTrue - The true values (labels) (shape: [n_samples, 1]).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients.
     */
    parameterGradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        const sampleCount = xTrue.shape[0];

        return tidy(() => {
            const errors = yPred.sub(yTrue);

            // Calculate the gradient of the MSE loss
            const biasGrad = errors.sum().div(sampleCount);
            // Calculate the gradient for each feature
            const weightGrad = xTrue.transpose().matMul(errors).div(sampleCount);

            // Concatenate bias gradient + feature gradients into one vector
            const gradients = concat([biasGrad.reshape([1, 1]), weightGrad]);

            return gradients as Tensor2D;
        });
    }

    /**
     * Computes the gradient of the Mean Squared Error (MSE) loss function with respect to the predictions.
     *
     * The gradient is calculated as:
     *   - grad = 2 * (y_pred - y_true)
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients of the MSE loss with respect to the predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return yPred.sub(yTrue).mul(2);
    }
}
