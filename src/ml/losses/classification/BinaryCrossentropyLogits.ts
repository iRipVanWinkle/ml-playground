import { type Scalar, type Tensor2D, exp, log1p, maximum, scalar, tidy } from '@tensorflow/tfjs';
import { BinaryCrossentropy } from './BinaryCrossentropy';

/**
 * Binary Crossentropy with Logits is a numerically stable version of binary crossentropy loss.
 *
 * Unlike standard binary crossentropy, this formulation directly operates on logits (raw scores before applying sigmoid).
 *
 * Formula:
 *     L(y_true, logits) = max(logits, 0) - logits * y_true + log(1 + exp(-|logits|))
 *
 * Advantages:
 * - Avoids numerical instability caused by very small or very large probabilities.
 * - Eliminates the need for clipping probabilities.
 *
 * Usage:
 * - Suitable for binary classification tasks where logits are used instead of probabilities.
 * - Commonly used in neural networks with sigmoid activation.
 */
export class BinaryCrossentropyLogits extends BinaryCrossentropy {
    usesLogits(): boolean {
        return true; // Indicates that this loss function uses logits directly
    }

    /**
     * Computes the Binary Crossentropy with Logits loss.
     *
     * This method calculates the loss directly from logits (raw scores before applying sigmoid),
     * ensuring numerical stability and eliminating the need for clipping probabilities.
     *
     * Formula:
     *     L(y_true, logits) = max(logits, 0) - logits * y_true + log(1 + exp(-|logits|))
     *
     * Explanation:
     * - `max(logits, 0)` handles positive logits.
     * - `log(1 + exp(-|logits|))` ensures numerical stability for large or small values.
     * - The loss is averaged over all samples to provide a single scalar value.
     *
     * @param yTrue - Tensor containing the true binary labels (0 or 1).
     * @param yPred - Tensor containing the predicted logits (raw scores before sigmoid).
     * @returns Scalar representing the Binary Crossentropy with Logits loss.
     */
    compute(yTrue: Tensor2D, logits: Tensor2D): Scalar {
        return tidy(() => {
            const zero = scalar(0);
            const maxZ = maximum(logits, zero); // max(z, 0)
            const logTerm = log1p(exp(logits.abs().neg())); // log(1 + exp(-|z|))

            // BCE: max(z, 0) - z * y + log(1 + exp(-|z|))
            const bce = maxZ.sub(logits.mul(yTrue)).add(logTerm).mean().asScalar();
            return bce as Scalar;
        });
    }
}
