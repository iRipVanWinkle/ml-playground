import { moments, scalar, tensor2d, tidy, type Tensor1D, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Applies z-score normalization.
 *
 * @param tensor - A tf.Tensor2D to be z-score scaled.
 * @returns A new tf.Tensor2D with z-score normalized values.
 */
export function zScoreScaling(tensor: Tensor2D): Tensor2D {
    return tidy(() => {
        if (tensor.size === 0) {
            return tensor2d([], [0, 0]);
        }

        // Compute mean and std deviation along axis 0 (per feature/column)
        const { mean, variance } = moments(tensor, 0);
        const std = variance.sqrt() as Tensor1D;

        // Prevent division by zero
        const safeStd = std.add(scalar(1e-8));

        // Center features
        const centered = tensor.sub(mean);

        // Normalize
        const scaled = centered.div(safeStd);

        return scaled as Tensor2D;
    });
}
