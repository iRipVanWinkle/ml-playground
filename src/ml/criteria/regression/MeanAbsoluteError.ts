import { MeanAbsoluteError as MeanAbsoluteErrorLoss } from '../../losses';
import { BaseFunction } from '../BaseFunction';

/**
 * Computes the median of a 2D tensor.
 *
 * @param array - The input tensor (shape: [n_samples, 1]).
 * @returns Scalar representing the median value.
 */
function computeMedian(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const n = sorted.length;
    const mid = Math.floor(n / 2);

    if (n % 2 === 1) {
        return sorted[mid];
    } else {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
}

export class MeanAbsoluteError extends BaseFunction {
    constructor(loss = new MeanAbsoluteErrorLoss()) {
        super(loss);
    }

    /**
     * Calculates the impurity score of a node's target values using Mean Absolute Error (mean absolute deviation).
     *
     * This method computes the mean absolute deviation of target values from their median
     * within a node. It measures the node's impurity based on absolute deviations, making
     * it more robust to outliers than variance.
     *
     * Formula:
     *     score = (1/n) * Σ |y_true - median(y_true)|
     *
     * where:
     *   - n: number of samples
     *   - y_true: true values (labels)
     *   - median(y_true): median value of all true values
     *
     * The median is used instead of mean because it minimizes MAE loss,
     * making it the theoretically optimal constant predictor for this metric.
     *
     * @param yTrue - The true values (labels) for the current node/dataset.
     * @returns number representing the baseline MAE score (MAD from median).
     */
    impurity(yTrue: number[][]): number {
        if (yTrue.length === 0) {
            return 0;
        }

        // Flatten the 2D array to get all target values
        const values = yTrue.flat();

        // Compute the median
        const median = computeMedian(values);

        // Calculate the variance (mean absolute deviation)
        const variance =
            values.reduce((sum, val) => {
                return sum + Math.abs(val - median);
            }, 0) / values.length;

        return variance;
    }
}
