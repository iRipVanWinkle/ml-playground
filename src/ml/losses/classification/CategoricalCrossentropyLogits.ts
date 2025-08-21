import { logSumExp, type Scalar, type Tensor2D, tidy } from '@tensorflow/tfjs';
import { CategoricalCrossentropy } from './CategoricalCrossentropy';

export class CategoricalCrossentropyLogits extends CategoricalCrossentropy {
    usesLogits(): boolean {
        return true; // Indicates that this loss function uses logits directly
    }

    /**
     * Computes the categorical cross-entropy loss for multiclass classification (using logits).
     *
     * The loss is computed as:
     *   L(y_true, logits) = log(sum(exp(logits))) - sum(y_true * logits)
     *
     * where:
     *   - y_true: one-hot encoded true labels (shape: [n_samples, n_classes])
     *   - logits: raw scores before softmax (shape: [n_samples, n_classes])
     *   - log(sum(exp(logits))) is computed using logSumExp for numerical stability
     * The loss is averaged over all samples.
     *
     * @param yTrueOneHot - The true one-hot encoded labels (shape: [n_samples, n_classes]).
     * @param yPred - The predicted logits (raw scores before softmax).
     * @returns Scalar representing the categorical cross-entropy loss with logits.
     */
    compute(yTrueOneHot: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => {
            // Compute log(sum(exp(logits))) using logSumExp for numerical stability
            // logSumExp prevents overflow/underflow that would occur with naive log(sum(exp(x)))
            // by using the mathematically equivalent: max(x) + log(sum(exp(x - max(x))))
            const logSum = logSumExp(yPred, 1);

            const crossEntropy = logSum.sub(yPred.mul(yTrueOneHot).sum(1));

            return crossEntropy.mean();
        });
    }
}
