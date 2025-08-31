import { MeanSquaredError as MeanSquaredErrorLoss } from '../../losses';
import { BaseFunction } from '../BaseFunction';

export class MeanSquaredError extends BaseFunction {
    constructor(loss = new MeanSquaredErrorLoss()) {
        super(loss);
    }

    /**
     * Calculates the impurity score of a node's target values using Mean Squared Error (variance).
     *
     * This method measures the variance (mean squared deviation) of the target values within
     * a single node. It is used in decision tree regression to evaluate how "pure" or
     * homogeneous the node is. A lower score indicates that the node's target values are
     * closer to their mean, meaning a better split.
     *
     * Formula:
     *     score = (1/n) * Σ(y_true - mean(y_true))²
     *
     * where:
     *   - n: number of samples
     *   - y_true: true values (labels)
     *   - mean(y_true): arithmetic mean of all true values
     *
     * @param yTrue - The true values (labels).
     * @returns Scalar representing the baseline MSE score (variance of true values).
     */
    impurity(yTrue: number[][]): number {
        if (yTrue.length === 0) {
            return 0;
        }

        // Flatten the 2D array to get all target values
        const values = yTrue.flat();

        // Calculate the mean
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;

        // Calculate the variance (mean squared error)
        const variance =
            values.reduce((sum, value) => {
                const diff = value - mean;
                return sum + diff * diff;
            }, 0) / values.length;

        return variance;
    }
}
