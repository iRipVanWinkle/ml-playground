import { type Scalar, type Tensor2D, concat, tidy } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { meanSquaredError } from '../../metrics';

export class MeanSquaredError implements LossFunction {
    /**
     * Mean Squared Error (MSE) is a common loss function used in regression tasks.
     *
     * It measures the average of the squares of the errors, which is the difference
     * between the predicted and actual values.
     *
     * Formula:
     *     MSE = (1/n) * Σ(y_true - y_pred)²
     *
     * where:
     *     - n: number of samples
     *     - y_true: true value
     *     - y_pred: predicted value
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the Mean Squared Error.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return meanSquaredError(yTrue, yPred);
    }

    /**
     * Computes the gradient of the Mean Squared Error (MSE) loss function.
     *
     * Formula:
     *   - For the bias term:
     *       ∇MSE_bias = (1/n) * Σ (y_pred - y_true)
     *   - For the weights:
     *       ∇MSE_weights = (1/n) * Σ (x * (y_pred - y_true))
     *
     * where:
     *   - n: number of samples
     *   - x: feature matrix
     *   - y_true: true values (labels)
     *   - y_pred: predicted values
     *
     * @param xTrue - The feature matrix.
     * @param yTrue - The true values (labels).
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
     * Formula:
     *   - grad = 2 * (y_pred - y_true)
     *
     * where:
     *   - y_true: true values (labels)
     *   - y_pred: predicted values
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return yPred.sub(yTrue).mul(2);
    }
}
