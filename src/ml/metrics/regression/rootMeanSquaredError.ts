import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { meanSquaredError } from './meanSquaredError';

/**
 * Computes the Root Mean Squared Error (RMSE) between true and predicted values.
 *
 * RMSE is the square root of MSE and provides an error metric in the same units
 * as the target variable.
 *
 * Formula:
 *     RMSE = √(MSE) = √((1/n) * Σ(y_true - y_pred)²)
 *
 * where:
 *   - n: number of samples
 *   - y_true: true values (labels)
 *   - y_pred: predicted values
 *
 * @param yTrue - The true target values.
 * @param yPred - The predicted target values.
 * @returns Scalar representing the Root Mean Squared Error.
 */
export function rootMeanSquaredError(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
    return tidy(() => meanSquaredError(yTrue, yPred).sqrt());
}
