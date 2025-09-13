import { concat, scalar, tidy, zeros, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Regularization } from '../types';
import { L1Regularization } from './l1';
import { L2Regularization } from './l2';

export class ElasticNetRegularization implements Regularization {
    private alpha: Scalar; // L1/L2 ratio parameter
    private alpha2D: Scalar;
    private zeros2D = zeros([1, 1]); // Used for bias term gradient

    private l1: L1Regularization;
    private l2: L2Regularization;

    constructor(lambda = 0, alpha = 0.5) {
        this.alpha = scalar(alpha); // alpha controls the L1/L2 ratio (0 = pure L2, 1 = pure L1)
        this.alpha2D = this.alpha.reshape([1, 1]); // Reshape for broadcasting

        this.l1 = new L1Regularization(lambda);
        this.l2 = new L2Regularization(lambda);
    }

    /**
     * Computes the ElasticNet regularization term.
     * ElasticNet = alpha * L1 + (1 - alpha) * L2
     * @param theta - The parameter vector (weights).
     * @returns The ElasticNet regularization term as a scalar.
     */
    compute(theta: Tensor2D): Scalar {
        return tidy(() => {
            // L1 regularization term: alpha * lambda * ||w||_1
            const l1Term = this.l1.compute(theta).mul(this.alpha);

            // L2 regularization term: (1 - alpha) * lambda * 0.5 * ||w||^2
            const l2Term = this.l2.compute(theta).mul(scalar(1).sub(this.alpha));

            // ElasticNet = L1 + L2
            return l1Term.add(l2Term);
        });
    }

    /**
     * Computes the gradient of the ElasticNet regularization term.
     * @param theta - The parameter vector (weights).
     * @returns The gradient of the ElasticNet regularization term.
     */
    gradient(theta: Tensor2D): Tensor2D {
        const [rows, cols] = theta.shape;

        return tidy(() => {
            // Create masks for L1 and L2 regularization
            const alphaMask = concat(
                [
                    this.zeros2D.tile([1, cols]), // Bias term gradient is 0 (no regularization)
                    this.alpha2D.tile([rows - 1, cols]), // L1 regularization for weights
                ],
                0,
            );

            const oneMinusAlphaMask = concat(
                [
                    this.zeros2D.tile([1, cols]), // Bias term gradient is 0 (no regularization)
                    scalar(1)
                        .sub(this.alpha2D)
                        .tile([rows - 1, cols]), // L2 regularization for weights
                ],
                0,
            );

            // L1 gradient: alpha * lambda * sign(w)
            const l1Gradient = this.l1.gradient(theta).mul(alphaMask);

            // L2 gradient: (1 - alpha) * lambda * w
            const l2Gradient = this.l2.gradient(theta).mul(oneMinusAlphaMask);

            // ElasticNet gradient = L1 gradient + L2 gradient
            return l1Gradient.add(l2Gradient);
        });
    }

    /**
     * Disposes the resources used by the regularization.
     */
    dispose(): void {
        this.alpha.dispose();
        this.alpha2D.dispose();
        this.zeros2D.dispose();
    }
}
