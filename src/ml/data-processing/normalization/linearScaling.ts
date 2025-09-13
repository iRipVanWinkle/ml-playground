import * as tf from '@tensorflow/tfjs';

/**
 * Linear scaling using TensorFlow.js.
 * Scales the input tensor to the [0, 1] range using min-max normalization.
 *
 * @param tensor - A tf.Tensor2D representing the matrix to be scaled.
 * @returns A new tf.Tensor2D with scaled values.
 */
export function linearScaling(tensor: tf.Tensor2D): tf.Tensor2D {
    if (tensor.size === 0) {
        return tf.tensor2d([], [0, 0]);
    }

    const min = tensor.min();
    const max = tensor.max();

    // Avoid division by zero if all values are the same
    const isConstant = tf.equal(min, max).arraySync();
    if (isConstant) {
        const zeros = tf.zerosLike(tensor);
        min.dispose();
        max.dispose();
        return zeros;
    }

    // Min-max scaling: (x - min) / (max - min)
    const scaled = tensor.sub(min).div(max.sub(min));
    min.dispose();
    max.dispose();

    return scaled as tf.Tensor2D;
}
