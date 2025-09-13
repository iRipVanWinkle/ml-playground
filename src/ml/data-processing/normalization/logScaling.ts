import { log, tensor2d, type Tensor2D } from '@tensorflow/tfjs';
/**
 * Applies logarithmic scaling to a tf.Tensor2D.
 * Computes the natural logarithm of each element after ensuring all values are positive.
 * Throws an error if any value is non-positive.
 *
 * @param tensor - A Tensor2D to be log-scaled.
 * @returns A new Tensor2D with log-scaled values.
 */
export function logScaling(tensor: Tensor2D): Tensor2D {
    if (tensor.size === 0) {
        return tensor2d([], [0, 0]);
    }

    const min = tensor.min();
    const minValue = min.arraySync() as number;
    if (minValue <= 0) {
        min.dispose();
        throw new Error(
            `Log scaling requires all values to be positive. Found minimum value: ${minValue}`,
        );
    }
    min.dispose();

    // Apply natural logarithm to each element
    return log(tensor);
}
