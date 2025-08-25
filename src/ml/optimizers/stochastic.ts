import { tensor1d, tidy, variable, type Tensor2D } from '@tensorflow/tfjs';
import { BaseOptimizer, type OptimizerOptions } from './base';
import type { OptimizeParameters } from '../types';
import { assert, range } from '../utils';

type StochasticOptimizerOptions = OptimizerOptions & {
    batchSize?: number;
};

const DEFAULT_BATCH_SIZE = 1;

export class StochasticGD extends BaseOptimizer {
    private batchSize: number;
    private batchIndexPool: number[] = [];
    private batchPoolPointer: number = 0;

    constructor(options: StochasticOptimizerOptions) {
        super(options);

        const { batchSize = DEFAULT_BATCH_SIZE } = options;

        assert(batchSize > 0, 'Batch size must be positive');

        this.batchSize = batchSize;
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
        initTheta,
    }: OptimizeParameters): Promise<Tensor2D> {
        const theta = variable(initTheta);

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
                const loss = lossFunction(batchX, batchY, theta);

                return loss;
            });

            const lossValue = (await loss.data())[0];

            // Dispose loss to free memory
            loss.dispose();
            batchX.dispose();
            batchY.dispose();

            await this.callback({ threadId, iteration, theta, loss: lossValue, alfa });

            // Check if the loss is NaN
            if (isNaN(lossValue)) {
                this.error(
                    `[${threadId}] Loss is NaN at iteration ${iteration}. Stopping optimization.`,
                );
                break;
            }
        }

        return theta;
    }

    private createBatch(X: Tensor2D, y: Tensor2D): [Tensor2D, Tensor2D] {
        const batchSize = this.batchSize;
        const sampleCount = X.shape[0];

        assert(batchSize <= sampleCount, 'Batch size cannot be larger than the number of samples.');

        // Refill and reshuffle when pointer exceeds pool
        if (this.batchPoolPointer + batchSize > sampleCount || this.batchIndexPool.length === 0) {
            this.refillBatchPool(sampleCount);
        }

        const indicesSlice = this.batchIndexPool.slice(
            this.batchPoolPointer,
            this.batchPoolPointer + batchSize,
        );
        this.batchPoolPointer += batchSize;

        const indices = tensor1d(indicesSlice, 'int32');

        const batchX = X.gather(indices);
        const batchY = y.gather(indices);

        indices.dispose();

        return [batchX, batchY];
    }

    private refillBatchPool(sampleCount: number): void {
        this.batchIndexPool = range(sampleCount);
        for (let i = sampleCount - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.batchIndexPool[i], this.batchIndexPool[j]] = [
                this.batchIndexPool[j],
                this.batchIndexPool[i],
            ];
        }
        this.batchPoolPointer = 0;
    }
}
