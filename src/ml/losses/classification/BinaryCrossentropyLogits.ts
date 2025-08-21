import { type Scalar, type Tensor2D, exp, log1p, maximum, scalar, tidy } from '@tensorflow/tfjs';
import { BinaryCrossentropy } from './BinaryCrossentropy';

export class BinaryCrossentropyLogits extends BinaryCrossentropy {
    usesLogits(): boolean {
        return true; // Indicates that this loss function uses logits directly
    }

    /**
     * Computes the binary cross-entropy loss for binary classification (using logits).
     *
     * The loss is computed as:
     *   L(y_true, logits) = max(logits, 0) - logits * y_true + log(1 + exp(-|logits|))
     *
     * where:
     *   - y_true: true binary labels (0 or 1) (shape: [n_samples, 1])
     *   - logits: raw scores before sigmoid (shape: [n_samples, 1])
     *   - log(1 + exp(-|logits|)) ensures numerical stability
     * The loss is averaged over all samples.
     *
     * @param yTrue - The true binary labels (shape: [n_samples, 1]).
     * @param logits - The predicted logits (raw scores before sigmoid).
     * @returns Scalar representing the binary cross-entropy loss with logits.
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
