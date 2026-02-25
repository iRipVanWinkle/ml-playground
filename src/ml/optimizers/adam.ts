import { tidy, variable, type Tensor2D } from '@tensorflow/tfjs';
import { BaseOptimizer, type OptimizerOptions } from './base';
import type { OptimizeParameters } from '../types';
import { assert } from '../utils';
import { EPSILON } from '../constants';

type AdamOptimizerOptions = OptimizerOptions & {
    beta1?: number;
    beta2?: number;
};

const DEFAULT_BETA1 = 0.9;
const DEFAULT_BETA2 = 0.999;

export class AdamGD extends BaseOptimizer {
    private beta1: number;
    private beta2: number;

    /**
     * Adam Gradient Descent Optimizer
     * @param beta1 - Exponential decay rate for the first moment estimates (default: 0.9)
     * @param beta2 - Exponential decay rate for the second moment estimates (default: 0.999)
     * @param epsilon - Small constant to prevent division by zero (default: 1e-7)
     * @param options - Optimizer options including learning rate, max iterations, tolerance, and regularization
     */
    constructor(options: AdamOptimizerOptions) {
        super(options);

        const { beta1 = DEFAULT_BETA1, beta2 = DEFAULT_BETA2 } = options;

        assert(
            beta1 > 0 && beta1 < 1,
            `Invalid beta1 value: ${beta1}. It should be in the range (0, 1).`,
        );
        assert(
            beta2 > 0 && beta2 < 1,
            `Invalid beta2 value: ${beta2}. It should be in the range (0, 1).`,
        );

        this.beta1 = beta1;
        this.beta2 = beta2;
    }
    /**
     * Optimizes the parameters using Adam Gradient Descent.
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
        // Initialize first and second moment vectors to zeros
        const m = tidy(() => variable(theta.zerosLike()));
        const v = tidy(() => variable(theta.zerosLike()));
        let t = 0; // Time step

        for await (const iteration of this.iterator()) {
            const alfa = this.learningRate.next(iteration);

            const loss = tidy(() => {
                // Compute the gradient
                const gradient = gradientFunction(X, y, theta);

                // Update time step
                t += 1;

                // Update first moment estimate
                m.assign(m.mul(this.beta1).add(gradient.mul(1 - this.beta1)));

                // Update second moment estimate
                v.assign(v.mul(this.beta2).add(gradient.square().mul(1 - this.beta2)));

                // Compute bias-corrected first moment estimate
                const mHat = m.div(1 - Math.pow(this.beta1, t));

                // Compute bias-corrected second moment estimate
                const vHat = v.div(1 - Math.pow(this.beta2, t));

                // Update theta using Adam update rule
                const nextTheta = theta.sub(
                    mHat.div(vHat.sqrt().add(EPSILON)).mul(alfa),
                ) as Tensor2D;

                theta.assign(nextTheta); // Assign the updated theta

                // Compute the loss with the updated theta
                const loss = lossFunction(X, y, theta);

                return loss;
            });

            const lossValue = (await loss.data())[0];

            await this.callback({ threadId, iteration, theta, loss: lossValue, alfa });

            loss.dispose(); // Dispose loss to free memory

            // Check if the loss is NaN
            if (isNaN(lossValue)) {
                this.error(
                    `[${threadId}] Loss is NaN at iteration ${iteration}. Stopping optimization.`,
                );
                break;
            }

            // If the loss is already below the tolerance, we can break early
            if (this.checkEarlyStopping(lossValue)) {
                this.info(
                    `[${threadId}] Early stopping at iteration ${iteration} with loss: ${lossValue}`,
                );
                break;
            }
        }

        // Dispose moment vectors to free memory
        m.dispose();
        v.dispose();

        return theta;
    }
}
