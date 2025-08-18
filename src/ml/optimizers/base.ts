import { Variable, variable, zeros, type Tensor2D, type Rank, tidy } from '@tensorflow/tfjs';
import { LearningRate } from '../LearningRate';
import type { OptimizeParameters, Optimizer } from '../types';
import { EventListener } from '../helpers/EventListener';
import { DEFAULT_TOLERANCE } from '../constants';

/**
 * Options for the optimizer.
 */
export type OptimizerOptions = Readonly<{
    learningRate: LearningRate | number;
    maxIterations: number;
    tolerance?: number;
    withBias?: boolean;
}>;

export abstract class BaseOptimizer extends EventListener implements Optimizer {
    protected learningRate: LearningRate;
    protected maxIterations: number;
    protected tolerance: number;
    protected withBias: boolean;

    private isPaused = false;
    private isStopped = false;
    private stepRequested = false;

    constructor(options: OptimizerOptions) {
        super();

        this.learningRate =
            options.learningRate instanceof LearningRate
                ? options.learningRate
                : new LearningRate(options.learningRate, 0, 0);
        this.maxIterations = options.maxIterations;
        this.tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
        this.withBias = options.withBias ?? true; // Default to true if not specified
    }

    stop(): void {
        this.isStopped = true;
    }

    pause(): void {
        this.isPaused = true;
    }

    resume(): void {
        this.isPaused = false;
    }

    step(): void {
        this.stepRequested = true;
    }

    async *iterator(): AsyncGenerator<number, void, unknown> {
        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            while (this.isPaused && !this.stepRequested) {
                await new Promise((resolve) => setTimeout(resolve, 100)); // Wait while paused
            }

            if (this.isStopped) {
                break; // Stop the generator
            }

            this.stepRequested = false; // Reset step request

            yield iteration; // Yield the current iteration
        }
    }

    abstract optimize(params: OptimizeParameters): Promise<Tensor2D>;

    protected inithTheta(X: Tensor2D): Variable<Rank.R2> {
        const featureCount = X.shape[1] + (this.withBias ? 1 : 0); // +1 for the bias term

        return tidy(() => variable(zeros([featureCount, 1])));
    }
}
