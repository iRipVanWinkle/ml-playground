import { tidy, variable, type Tensor2D } from '@tensorflow/tfjs';
import { BaseOptimizer, type OptimizerOptions } from './base';
import type { OptimizeParameters } from '../types';
import { assert } from '../utils';

type MomentumOptimizerOptions = OptimizerOptions & {
    beta?: number;
};

const DEFAULT_BETA = 0.9;

export class MomentumGD extends BaseOptimizer {
    private beta: number;

    /**
     * Momentum Gradient Descent Optimizer
     * @param beta - Momentum factor (default: 0.9)
     * @param options - Optimizer options including learning rate, max iterations, tolerance, and regularization
     */
    constructor(options: MomentumOptimizerOptions) {
        super(options);

        const { beta = DEFAULT_BETA } = options;

        assert(
            beta > 0 && beta < 1,
            `Invalid beta value: ${beta}. It should be in the range (0, 1).`,
        );

        this.beta = beta;
    }
    /**
     * Optimizes the parameters using Momentum Gradient Descent.
     * @param lossFunction - Function to compute the loss given the parameters.
     * @param gradientFunction - Function to compute the gradient of the loss with respect to the parameters.
     * @param initTheta - Initial parameters (weights).
     * @returns Optimized parameters (weights).
     */
    async optimize({
        X,
        y,
        lossFunction,
        gradientFunction,
        threadId = 0,
        initTheta,
    }: OptimizeParameters): Promise<Tensor2D> {
        const theta = variable(initTheta);

        // Initialize velocity to zeros
        const velocity = tidy(() => variable(theta.zerosLike()));

        for await (const iteration of this.iterator()) {
            const alfa = this.learningRate.next(iteration);

            const loss = tidy(() => {
                // Compute the gradient
                const gradient = gradientFunction(X, y, theta);

                // Update velocity
                const nextVelocity = velocity.mul(this.beta).sub(gradient.mul(alfa)) as Tensor2D;
                velocity.assign(nextVelocity);

                // Update theta using the velocity
                const nextTheta = theta.add(velocity) as Tensor2D;

                theta.assign(nextTheta); // Assign the updated theta

                // Compute the loss with the updated theta
                const loss = lossFunction(X, y, theta);

                return loss;
            });

            const lossValue = (await loss.data())[0];

            loss.dispose(); // Dispose loss to free memory

            await this.callback({ threadId, iteration, theta, loss: lossValue, alfa });

            // Check if the loss is NaN
            if (isNaN(lossValue)) {
                this.error(
                    `[${threadId}] Loss is NaN at iteration ${iteration}. Stopping optimization.`,
                );
                break;
            }

            // If the loss is already below the tolerance, we can break early
            if (lossValue < this.tolerance) {
                this.info(
                    `[${threadId}] Early stopping at iteration ${iteration} with loss: ${lossValue.toFixed(4)}`,
                );
                break;
            }
        }

        return theta;
    }
}
