import { concat, ones, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { LossFunction, Optimizer, Model, Regularization } from '../../types';
import { NoRegularization } from '../../regularization';

export type ModelOptions = {
    lossFunc: LossFunction;
    optimizer: Optimizer;
    regularization?: Regularization;
};

export abstract class BaseEstimator implements Model<Tensor2D> {
    protected lossFunc: LossFunction;
    protected optimizer: Optimizer;
    protected regularization: Regularization;

    protected theta: Tensor2D | null = null;

    constructor(options: ModelOptions) {
        this.lossFunc = options.lossFunc;
        this.optimizer = options.optimizer;
        this.regularization = options.regularization ?? new NoRegularization();
    }

    abstract train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D>;

    abstract predict(X: Tensor2D, theta?: Tensor2D): Tensor2D;

    abstract evaluate(X: Tensor2D, y: Tensor2D, theta?: Tensor2D): [Tensor2D, Tensor2D, Scalar];

    getTheta(): Tensor2D | null {
        return this.theta;
    }

    dispose(withDependencies = false): void {
        this.theta?.dispose();

        if (withDependencies) {
            this.lossFunc.dispose?.();
            this.optimizer.dispose?.();
            this.regularization.dispose?.();
        }
    }

    stop(): void {
        this.optimizer.stop();
    }

    pause(): void {
        this.optimizer.pause();
    }

    step(): void {
        this.optimizer.step();
    }

    resume(): void {
        this.optimizer.resume();
    }

    protected addBiasTerm(X: Tensor2D): Tensor2D {
        return tidy(() => concat([ones([X.shape[0], 1]), X], 1) as Tensor2D);
    }
}
