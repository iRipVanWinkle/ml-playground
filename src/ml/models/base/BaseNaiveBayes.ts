import type { Tensor2D, Scalar } from '@tensorflow/tfjs';
import type { Model, NaiveBayesParams, TrainingControl, TrainingEventEmitter } from '../../types';

export type BaseNaiveBayesOptions = {
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

export abstract class BaseNaiveBayes<T extends NaiveBayesParams> implements Model<T> {
    protected params?: T;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;

    constructor(options: BaseNaiveBayesOptions) {
        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;
    }

    abstract train(X: Tensor2D, y: Tensor2D): Promise<T>;
    abstract predict(X: Tensor2D, theta?: T | undefined): Tensor2D;
    abstract evaluate(
        X: Tensor2D,
        y: Tensor2D,
        theta?: T | undefined,
    ): [Tensor2D, Tensor2D, Scalar];

    dispose(): void {
        this.params = undefined;
    }
}
