import { log, tensor2d, tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { Scaler, LogScalerParams } from '@/ml/types';

/**
 * Applies logarithmic scaling to a tf.Tensor2D.
 * Computes the natural logarithm of each element after ensuring all values are positive.
 * Throws an error if any value is non-positive.
 */
export class LogScaler implements Scaler<LogScalerParams> {
    fit(): void {
        // No parameters to fit
    }

    /**
     * Transforms the input tensor using log scaling.
     *
     * @param tensor - A Tensor2D to be log-scaled.
     * @returns A new Tensor2D with log-scaled values.
     */
    transform(tensor: Tensor2D): Tensor2D {
        if (tensor.size === 0) {
            return tensor2d([], [0, 0]);
        }

        const min = tensor.min();
        const minValue = min.arraySync() as number;
        min.dispose();

        if (minValue <= 0) {
            throw new Error(
                `Log scaling requires all values to be positive. Found minimum value: ${minValue}`,
            );
        }

        // Apply natural logarithm to each element
        return tidy(() => log(tensor));
    }
}
