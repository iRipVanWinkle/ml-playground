import { LogCosh as LogCoshLoss } from '../../losses';
import { BaseFunction } from '../BaseFunction';

export class LogCosh extends BaseFunction {
    constructor(loss = new LogCoshLoss()) {
        super(loss);
    }

    /**
     * Calculates the impurity score of a node's target values using Log-Cosh loss.
     *
     * This method computes the average of log(cosh(y - mean)) over the node's target values,
     * providing a smooth, robust measure of impurity that behaves similarly to MSE near zero
     * but is less sensitive to outliers.
     *
     * Formula:
     *     score = (1/n) * Σ log(cosh(y_true - mean(y_true)))
     *
     * where:
     *   - n: number of samples
     *   - y_true: true values (labels)
     *   - mean(y_true): arithmetic mean of all true values
     *
     * @param yTrue - The true values (labels).
     * @returns Scalar representing the baseline Log-Cosh loss score.
     */
    impurity(yTrue: number[][]): number {
        if (yTrue.length === 0) {
            return 0;
        }

        // Flatten the 2D array to get all target values
        const values = yTrue.flat();

        // Calculate the mean
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;

        // Calculate log-cosh loss: log(cosh(y - mean))
        const logCoshSum = values.reduce((sum, value) => {
            const diff = value - mean;
            // cosh(x) = (e^x + e^(-x)) / 2
            const coshValue = (Math.exp(diff) + Math.exp(-diff)) / 2;
            return sum + Math.log(coshValue);
        }, 0);

        return logCoshSum / values.length;
    }
}
