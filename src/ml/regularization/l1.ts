import { concat, scalar, sign, tidy, zeros, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Regularization } from '../types';

export class L1Regularization implements Regularization {
    private lambda: Scalar;
    private lambda2D: Scalar;
    private zeros2D = zeros([1, 1]); // Used for bias term gradient

    constructor(lambda = 0) {
        this.lambda = scalar(lambda);
        this.lambda2D = this.lambda.reshape([1, 1]); // Reshape for broadcasting
    }

    /**
     * Computes the L1 regularization term.
     * @param theta - The parameter vector (weights).
     * @returns The L1 regularization term as a scalar.
     */
    compute(theta: Tensor2D): Scalar {
        return tidy(() => {
            const weights = theta.slice([1, 0]); // Exclude the bias term, keep only weights

            // L1 regularization term: lambda * ||w||_1
            return this.lambda.mul(weights.abs().sum());
        });
    }

    /**
     * Computes the gradient of the L1 regularization term.
     * @param theta - The parameter vector (weights).
     * @returns The gradient of the L1 regularization term.
     */
    gradient(theta: Tensor2D): Tensor2D {
        const [rows, cols] = theta.shape;

        return tidy(() => {
            // The first row is for the bias term, which does not have regularization
            // Bias term gradient is 0, regularization for weights is lambda
            // Create a mask for the gradient: [0, lambda, lambda, ..., lambda]
            const lambdaMask = concat(
                [
                    this.zeros2D.tile([1, cols]), // Bias term gradient is 0 (no regularization)
                    this.lambda2D.tile([rows - 1, cols]), // Regularization for weights
                ],
                0,
            );

            // Gradient of L1 regularization: lambda * sign(w)
            return lambdaMask.mul(sign(theta)) as Tensor2D;
        });
    }

    /**
     * Disposes the resources used by the regularization.
     */
    dispose(): void {
        this.lambda.dispose();
        this.lambda2D.dispose();
        this.zeros2D.dispose();
    }
}
