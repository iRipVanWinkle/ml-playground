import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { EPSILON } from '../../constants';

/**
 * Computes the R² (coefficient of determination) regression score.
 *
 * R² represents the proportion of variance in the dependent variable that is predictable
 * from the independent variable(s). It ranges from -∞ to 1, where 1 indicates perfect
 * prediction, 0 indicates that the model performs no better than a simple mean,
 * and negative values indicate worse performance than a mean model.
 *
 * Formula:
 *     R² = 1 - (SS_res / SS_tot)
 * where:
 *     - SS_res: residual sum of squares = Σ(y_true - y_pred)²
 *     - SS_tot: total sum of squares = Σ(y_true - mean(y_true))²
 *
 * @param yTrue - The true values (labels).
 * @param yPred - The predicted values.
 * @returns Scalar representing the R² score.
 */
export function r2Score(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
    return tidy(() => {
        // Calculate residual sum of squares: Σ(y_true - y_pred)²
        const residuals = yTrue.sub(yPred);
        const ssRes = residuals.square().sum();

        // Calculate total sum of squares: Σ(y_true - mean(y_true))²
        const yMean = yTrue.mean();
        const totalDeviation = yTrue.sub(yMean);
        const ssTot = totalDeviation.square().sum();

        // Calculate R² = 1 - (SS_res / SS_tot)
        return ssRes.div(ssTot.add(EPSILON)).mul(-1).add(1) as Scalar;
    });
}
