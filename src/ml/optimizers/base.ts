import { getBackend, type Tensor2D } from '@tensorflow/tfjs';
import { LearningRate } from '../LearningRate';
import type {
    OptimizeParameters,
    Optimizer,
    OptimizerCallbackParameters,
    TrainingControl,
    TrainingEventEmitter,
} from '../types';
import { DEFAULT_TOLERANCE } from '../constants';
import { assert } from '../utils';

/**
 * Options for the optimizer.
 */
export type OptimizerOptions = Readonly<{
    learningRate: LearningRate | number;
    maxIterations: number;
    tolerance?: number;
    withBias?: boolean;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
}>;

export abstract class BaseOptimizer implements Optimizer {
    protected learningRate: LearningRate;
    protected maxIterations: number;
    protected tolerance: number;
    protected withBias: boolean;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;

    constructor(options: OptimizerOptions) {
        const {
            learningRate,
            maxIterations,
            eventEmitter,
            trainingController,
            tolerance = DEFAULT_TOLERANCE,
            withBias = true,
        } = options;

        assert(maxIterations > 0, 'Max iterations must be positive');
        assert(tolerance > 0, 'Tolerance must be positive');

        this.learningRate =
            learningRate instanceof LearningRate
                ? learningRate
                : new LearningRate(learningRate, 0, 0);

        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.withBias = withBias;
        this.eventEmitter = eventEmitter;
        this.trainingController = trainingController;
    }

    async *iterator(): AsyncGenerator<number, void, unknown> {
        const isSyncBackend = this.isSyncBackend();

        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            await this.trainingController?.handleControlFlow(isSyncBackend);

            if (this.trainingController?.isTrainingStopped) {
                break; // Stop the generator
            }

            yield iteration; // Yield the current iteration
        }
    }

    abstract optimize(params: OptimizeParameters): Promise<Tensor2D>;

    protected info(message: string): void {
        this.eventEmitter?.emit('info', message);
    }

    protected error(message: string): void {
        this.eventEmitter?.emit('error', message);
    }

    protected callback(params: OptimizerCallbackParameters): Promise<void> | undefined {
        return this.eventEmitter?.emit('callback', params);
    }

    private isSyncBackend(): boolean {
        return getBackend() === 'cpu' || getBackend() === 'wasm';
    }
}
