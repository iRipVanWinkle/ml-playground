import { tensor1d, tidy, util, type Tensor2D } from '@tensorflow/tfjs';
import { BaseOptimizer, type OptimizerOptions } from './base';
import type { OptimizeParameters } from '../types';

type StochasticOptimizerOptions = OptimizerOptions & {
    batchSize?: number;
};

const DEFAULT_BATCH_SIZE = 1;

export class StochasticGD extends BaseOptimizer {
    private batchSize: number;

    constructor(options: StochasticOptimizerOptions) {
        super(options);

        this.batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
    }
    /**
     * Optimizes the model parameters using Stochastic Gradient Descent.
     * @param lossFunction - The loss function to minimize.
     * @param gradientFunction - The function to compute the gradient of the loss.
     * @param initTheta - Initial model parameters.
     * @param originalData - The complete dataset for training.
     * @returns The optimized model parameters.
     */
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

            // Sample a random batch
            const [batchX, batchY] = this.createBatch(X, y);

            const loss = tidy(() => {
                // Compute the gradient
                const gradient = gradientFunction(batchX, batchY, theta);

                // Update theta using the gradient
                const nextTheta = theta.sub(gradient.mul(alfa)) as Tensor2D;

                // Update theta
                theta.assign(nextTheta);

                // Compute the loss
                const loss = lossFunction(X, y, theta);

                return loss;
            });

            await this.emit('callback', { threadId, iteration, theta, loss, alfa });

            const lossValue = (await loss.data())[0];

            // Dispose loss to free memory
            loss.dispose();
            batchX.dispose();
            batchY.dispose();

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

    private createBatch(X: Tensor2D, y: Tensor2D): [Tensor2D, Tensor2D] {
        const batchSize = this.batchSize;
        const sampleCount = X.shape[0];

        if (batchSize > sampleCount) {
            throw new Error('Batch size cannot be larger than the number of samples.');
        }

        const indices = tensor1d(
            Array.from(util.createShuffledIndices(sampleCount).slice(0, batchSize)),
            'int32',
        );
        const batchX = X.gather(indices);
        const batchY = y.gather(indices);

        indices.dispose(); // Dispose of indices to free memory

        return [batchX, batchY];
    }
}
