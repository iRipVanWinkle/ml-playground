import { concat, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { LossFunction } from '../types';

export class MeanAbsoluteError implements LossFunction {
    /**
     * Mean Absolute Error (MAE) is a loss function used in regression tasks.
     *
     * It measures the average of the absolute differences between predicted and actual values.
     *
     * Formula:
     *     MAE = (1/n) * Σ |y_true - y_pred|
     *
     * where:
     *     - n: number of samples
     *     - y_true: true values (labels)
     *     - y_pred: predicted values
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => yPred.sub(yTrue).abs().mean());
    }

    /**
     * Computes the gradient of the Mean Absolute Error (MAE) loss function.
     *
     * The gradients are computed as follows:
     *   - For the bias term:
     *       ∇MAE_bias = (1/n) * Σ sign(y_pred - y_true)
     *   - For the weights:
     *       ∇MAE_weights = (1/n) * Σ (sign(y_pred - y_true) * x)
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
    gradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        const sampleCount = xTrue.shape[0];

        return tidy(() => {
            const errors = yPred.sub(yTrue);
            const signErrors = errors.sign();

            const biasGrad = signErrors.sum().div(sampleCount);

            const weightGrad = xTrue.transpose().matMul(signErrors).div(sampleCount);

            // Concatenate bias gradient + feature gradients into one vector
            const gradients = concat([biasGrad.reshape([1, 1]), weightGrad]);

            return gradients as Tensor2D;
        });
    }
}
