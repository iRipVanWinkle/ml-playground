import { tidy, type Tensor2D, sub } from '@tensorflow/tfjs';

/**
 * Computes residuals (errors) between true and predicted values.
 *
 * Residuals are the differences between observed and predicted values:
 *     residual = y_true - y_pred
 *
 * where:
 *   - y_true: true values (labels)
 *   - y_pred: predicted values
 *
 * @param yTrue - The true values (labels).
 * @param yPred - The predicted values.
 * @returns Residuals (errors) between true and predicted values.
 */
export function residuals(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D {
    return tidy(() => sub(yTrue, yPred));
}
