import { concat, ones, tensor2d, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type {
    LossFunction,
    Optimizer,
    Model,
    Regularization,
    PredictionMetadata,
} from '../../types';
import { NoRegularization } from '../../regularization';
import {
    zerosInitializer,
    type ThetaInitializer,
} from '../../factories/theta-initialization/initializers';
import { getMatrixFromTensor, type MatrixLike } from '../../matrix';
import { assertModelTrained } from '../../utils';

export type ModelOptions = {
    lossFunc: LossFunction;
    optimizer: Optimizer;
    regularization?: Regularization;
    thetaInitializer?: ThetaInitializer;
};

export abstract class BaseEstimator implements Model<Tensor2D> {
    protected lossFunc: LossFunction;
    protected optimizer: Optimizer;
    protected regularization: Regularization;
    protected thetaInitializer: ThetaInitializer;

    protected theta: Tensor2D | null = null;

    constructor(options: ModelOptions) {
        this.lossFunc = options.lossFunc;
        this.optimizer = options.optimizer;
        this.thetaInitializer = options.thetaInitializer ?? zerosInitializer();
        this.regularization = options.regularization ?? new NoRegularization();
    }

    abstract train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D>;

    abstract predict(X: Tensor2D, theta?: Tensor2D): Tensor2D;

    abstract evaluate(X: Tensor2D, y: Tensor2D, theta?: Tensor2D): [Tensor2D, Tensor2D, Scalar];

    abstract predictWithMetadata(X: Tensor2D, theta?: Tensor2D): PredictionMetadata;

    async extractParameters(): Promise<MatrixLike> {
        assertModelTrained(this.theta);

        return getMatrixFromTensor(this.theta);
    }

    restoreParameters(params: MatrixLike): void {
        this.theta = tensor2d(params.array, params.shape);
    }

    dispose(withDependencies = false): void {
        this.theta?.dispose();

        if (withDependencies) {
            this.lossFunc.dispose?.();
            this.optimizer.dispose?.();
            this.regularization.dispose?.();
        }
    }

    protected addBiasTerm(X: Tensor2D): Tensor2D {
        return tidy(() => concat([ones([X.shape[0], 1]), X], 1) as Tensor2D);
    }
}
