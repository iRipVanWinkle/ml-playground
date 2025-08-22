import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import { BaseOptimizer } from './base';
import type { OptimizeParameters } from '../types';

export class BatchGD extends BaseOptimizer {
    async optimize({
        X,
        y,
        lossFunction,
        gradientFunction,
        threadId = 0,
        inithThetaFunction = this.inithTheta.bind(this),
    }: OptimizeParameters): Promise<Tensor2D> {
        const theta = inithThetaFunction(X);

        for await (const iteration of this.iterator()) {
            const alfa = this.learningRate.next(iteration);

            const loss = tidy(() => {
                // Compute the gradient
                const gradient = gradientFunction(X, y, theta);
                // Update theta using the gradient
                const nextTheta = theta.sub(gradient.mul(alfa)) as Tensor2D;

                // Update theta
                theta.assign(nextTheta);

                // Compute the loss
                const loss = lossFunction(X, y, theta);

                return loss;
            });

            const lossValue = (await loss.data())[0];

            loss.dispose(); // Dispose loss to free memory

            await this.emit('callback', { threadId, iteration, theta, loss: lossValue, alfa });

            // Check if the loss is NaN
            if (isNaN(lossValue)) {
                this.emit(
                    'error',
                    `[${threadId}] Loss is NaN at iteration ${iteration}. Stopping optimization.`,
                );
                break;
            }

            // If the loss is already below the tolerance, we can break early
            if (lossValue < this.tolerance) {
                this.emit(
                    'info',
                    `[${threadId}] Early stopping at iteration ${iteration} with loss: ${lossValue}`,
                );
                break;
            }
        }

        return theta;
    }
}
