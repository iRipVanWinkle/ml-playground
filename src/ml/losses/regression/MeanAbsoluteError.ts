import {
    concat,
    onesLike,
    scalar,
    tidy,
    where,
    zerosLike,
    type Scalar,
    type Tensor2D,
} from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';

export class MeanAbsoluteError implements LossFunction {
    /**
     * Computes the Mean Absolute Error (MAE) loss function.
     *
     * The loss is computed as:
     *   MAE = (1/n) * Σ |y_true - y_pred|
     *
     * where:
     *   - n: number of samples
     *   - y_true: true values (labels) (shape: [n_samples, 1])
     *   - y_pred: predicted values (shape: [n_samples, 1])
     *
     * @param yTrue - The true values (labels) (shape: [n_samples, 1]).
     * @param yPred - The predicted values (shape: [n_samples, 1]).
     * @returns Scalar representing the mean absolute error.
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
     *   - x: feature matrix (shape: [n_samples, n_features])
     *   - y_true: true values (labels) (shape: [n_samples, 1])
     *   - y_pred: predicted values (shape: [n_samples, 1])
     *
     * @param xTrue - The feature matrix (shape: [n_samples, n_features]).
     * @param yTrue - The true values (labels) (shape: [n_samples, 1]).
     * @param yPred - The predicted values (shape: [n_samples, 1]).
     * @returns Tensor2D containing the gradients.
     */
    parameterGradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
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

    /**
     * Computes the gradient of the Mean Absolute Error (MAE) loss function with respect to the predictions.
     *
     * The gradient is computed as follows:
     *   - If y_pred > y_true, gradient = 1
     *   - If y_pred < y_true, gradient = -1
     *   - If y_pred == y_true, gradient = 0
     *
     * where:
     *   - y_true: true values (labels) (shape: [n_samples, 1])
     *   - y_pred: predicted values (shape: [n_samples, 1])
     *
     * @param yTrue - The true values (labels) (shape: [n_samples, 1]).
     * @param yPred - The predicted values (shape: [n_samples, 1]).
     * @returns Tensor2D containing the gradients.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return tidy(() => {
            return where(
                yPred.greater(yTrue),
                onesLike(yTrue),
                where(yPred.less(yTrue), scalar(-1).mul(onesLike(yTrue)), zerosLike(yTrue)),
            ) as Tensor2D;
        });
    }
}
