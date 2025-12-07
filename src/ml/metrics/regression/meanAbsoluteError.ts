import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Computes the Mean Absolute Error (MAE) between true and predicted values.
 *
 * MAE measures the average absolute difference between predicted and actual values.
 * It is more robust to outliers compared to MSE as it doesn't square the errors.
 *
 * Formula:
 *     MAE = (1/n) * Σ|y_true - y_pred|
 *
 * where:
 *   - n: number of samples
 *   - y_true: true values (labels)
 *   - y_pred: predicted values
 *
 * @param yTrue - The true values (labels).
 * @param yPred - The predicted values.
 * @returns Scalar representing the mean absolute error.
 */
export function meanAbsoluteError(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
    return tidy(() => yPred.sub(yTrue).abs().mean());
}
