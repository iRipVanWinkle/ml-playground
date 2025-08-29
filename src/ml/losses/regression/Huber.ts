import { concat, neg, scalar, tidy, where, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';

export class Huber implements LossFunction {
    delta: Scalar;

    constructor(delta: number = 1.0) {
        this.delta = scalar(delta);
    }

    /**
     * Huber loss is a loss function used in regression tasks that is less sensitive to outliers than the squared error loss.
     * It combines the properties of Mean Squared Error (MSE) and Mean Absolute Error (MAE).
     *
     * Formula:
     *     L(y_true, y_pred) =
     *         0.5 * (y_true - y_pred)²,           if |y_true - y_pred| ≤ δ
     *         δ * |y_true - y_pred| - 0.5 * δ²,   otherwise
     *
     * where:
     *     - δ (delta): threshold parameter that determines the point where the loss function changes from quadratic to linear
     *     - y_true: true value (label)
     *     - y_pred: predicted value
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the Huber loss.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        const delta = this.delta;

        return tidy(() => {
            const error = yPred.sub(yTrue);
            const absError = error.abs();

            const quadraticPart = error.square().div(2);
            const linearPart = absError.mul(delta).sub(delta.square().div(2));

            const useQuadratic = absError.lessEqual(delta);

            const loss = where(useQuadratic, quadraticPart, linearPart);
            return loss.mean();
        });
    }

    /**
     * Computes the gradient of the Huber loss function with respect to the model parameters.
     *
     * The gradients are computed as follows:
     *   - For the bias term:
     *       ∇L_bias = Σ grad(y_pred - y_true)
     *   - For the weights:
     *       ∇L_weights = Σ (grad(y_pred - y_true) * x)
     *
     * where:
     *   - grad(y_pred - y_true) =
     *         error,                  if |y_pred - y_true| ≤ δ
     *         δ * sign(y_pred - y_true), otherwise
     *   - n: number of samples
     *   - x: feature matrix
     *   - y_true: true values (labels)
     *   - y_pred: predicted values
     *   - δ (delta): threshold parameter
     *
     * @param xTrue - The feature matrix (shape: [n_samples, n_features]).
     * @param yTrue - The true values (labels) (shape: [n_samples, 1]).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients of the Huber loss with respect to the model parameters.
     */
    parameterGradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        const delta = this.delta;

        return tidy(() => {
            const error = yPred.sub(yTrue);
            const absError = error.abs();

            const isSmallError = absError.lessEqual(delta); // boolean mask

            const gradSmall = error; // if |error| <= delta, use error
            const gradLarge = error.sign().mul(delta); // else, delta * sign(error)

            const grad = where(isSmallError, gradSmall, gradLarge);

            const biasGrad = grad.sum().reshape([1, 1]);
            const weightGrad = xTrue.transpose().matMul(grad);

            return concat([biasGrad, weightGrad]) as Tensor2D;
        });
    }

    /**
     * Computes the gradient of the Huber loss function with respect to the predictions.
     *
     * The gradient is calculated as:
     *   - grad =
     *         error,                  if |error| ≤ δ
     *         δ * sign(error),       otherwise
     *
     * where:
     *   - error = y_pred - y_true
     *   - δ (delta): threshold parameter
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients of the Huber loss with respect to the predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
        const delta = this.delta;

        return tidy(() => {
            const error = yPred.sub(yTrue);
            const absError = error.abs();

            const isSmallError = absError.lessEqual(delta);

            const gradSmall = error;
            const gradLarge = where(error.greater(0), delta, neg(delta));

            return where(isSmallError, gradSmall, gradLarge) as Tensor2D;
        });
    }

    /**
     * Disposes of the resources used by the HuberLoss instance.
     */
    dispose(): void {
        this.delta.dispose();
    }
}
