import { type Tensor2D, type Scalar, concat, tidy, tanh } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';

export class LogCosh implements LossFunction {
    /**
     * Log-Cosh loss is a smooth loss function used in regression tasks.
     *
     * It is calculated as the mean of log(cosh(y_pred - y_true)) for all samples.
     * Log-Cosh behaves similarly to Mean Squared Error (MSE) for small differences,
     * but is less sensitive to large outliers, like Mean Absolute Error (MAE).
     *
     * Formula:
     *     logcosh(x) = log((e^x + e^(-x)) / 2)
     *     Loss = (1/n) * Σ logcosh(y_true - y_pred)
     *
     * where:
     *   - n: number of samples
     *   - y_true: true values (labels)
     *   - y_pred: predicted values
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the Log-Cosh loss.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => yPred.sub(yTrue).cosh().log().mean());
    }

    /**
     * Computes the gradient of the Log-Cosh loss function with respect to the model parameters.
     *
     * The gradient of Log-Cosh loss is given by:
     *   - grad = tanh(y_pred - y_true)
     *
     * The gradients are computed as follows:
     *   - For the bias term:
     *       ∇L_bias = mean(tanh(y_pred - y_true))
     *   - For the weights:
     *       ∇L_weights = (1/n) * Σ [x * tanh(y_pred - y_true)]
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
            const diff = yPred.sub(yTrue);
            const grad = diff.tanh();

            const biasGrad = grad.mean();
            const weightGrad = xTrue.transpose().matMul(grad).div(sampleCount);

            const gradients = concat([biasGrad.reshape([1, 1]), weightGrad]);

            return gradients as Tensor2D;
        });
    }

    /**
     * Computes the gradient of the Log-Cosh loss function with respect to the predictions.
     *
     * The gradient is calculated as:
     *   - grad = tanh(y_pred - y_true)
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Tensor2D representing the gradient of the Log-Cosh loss with respect to the predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        return tanh(yPred.sub(yTrue)) as Tensor2D;
    }
}
