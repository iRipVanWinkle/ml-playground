import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Computes the Mean Squared Error (MSE) between true and predicted values.
 *
 * It measures the average of the squares of the errors, which is the difference
 * between the predicted and actual values.
 *
 * Formula:
 *     MSE = (1/n) * Σ(y_true - y_pred)²
 *
 * where:
 *   - n: number of samples
 *   - y_true: true values (labels)
 *   - y_pred: predicted values
 *
 * @param yTrue - The true target values.
 * @param yPred - The predicted target values.
 * @returns Scalar representing the Mean Squared Error.
 */
export function meanSquaredError(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
    return tidy(() => yPred.sub(yTrue).square().mean());
}
