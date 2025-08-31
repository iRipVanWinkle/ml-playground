import { scalar, type Scalar } from '@tensorflow/tfjs';
import { Huber as HuberLoss } from '../../losses';
import { BaseFunction } from '../BaseFunction';

export class Huber extends BaseFunction {
    private delta: Scalar;

    constructor(delta = 1.0, loss = new HuberLoss(delta)) {
        super(loss);
        this.delta = scalar(delta);
    }

    impurity(yTrue: number[][]): number {
        if (yTrue.length === 0) {
            return 0;
        }

        // Flatten the 2D array to get all target values
        const values = yTrue.flat();

        // Calculate the mean
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;

        // Get delta value (need to extract from the Scalar tensor)
        const deltaValue = this.delta.dataSync()[0];

        // Calculate Huber loss for each value
        let totalLoss = 0;
        for (let i = 0; i < values.length; i++) {
            const error = Math.abs(values[i] - mean);

            if (error <= deltaValue) {
                // Quadratic part: 0.5 * error^2
                totalLoss += 0.5 * error * error;
            } else {
                // Linear part: delta * |error| - 0.5 * delta^2
                totalLoss += deltaValue * error - 0.5 * deltaValue * deltaValue;
            }
        }

        return totalLoss / values.length;
    }

    /**
     * Disposes of the resources used by the HuberLoss instance.
     */
    dispose(): void {
        this.delta.dispose();
        super.dispose();
    }
}
