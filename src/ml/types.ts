import type { Rank, Scalar, Tensor2D, Tensor3D, Variable } from '@tensorflow/tfjs';
import type { EventEmitter } from './events/EventEmitter';
import type { MatrixLike } from './matrix';

export type Variable2D = Variable<Rank.R2>;

/**
 * Computes the metric between true values and predicted values.
 *
 * @param yTrue - The true values (labels).
 * @param yPred - The predicted values.
 * @returns Scalar representing the computed metric.
 */
export type MetricFunction = (yTrue: Tensor2D, yPred: Tensor2D) => Scalar;

/**
 * Aggregates the predictions from multiple trees in the ensemble.
 *
 * @param predictions - The predictions from the individual trees.
 * @returns The aggregated predictions.
 */
export type EnsembleAggregatorFn = (predictions: Tensor2D | Tensor3D) => Tensor2D;

export type TreeCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    tree: TreeNode;
    threadName?: string;
}>;

export type OptimizerCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    alfa: number;
    loss: number;
    theta: Tensor2D;
    threadName?: string;
}>;

export type NaiveBayesCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    params: NaiveBayesParams;
}>;

export type CallbackParameters =
    | OptimizerCallbackParameters
    | TreeCallbackParameters
    | NaiveBayesCallbackParameters;

export type TrainingState = 'transforming' | 'training' | 'paused' | 'stopped' | 'stepped-forward';
/**
 * Interface for training event listeners.
 */
export interface TrainingEventEmitter extends EventEmitter {
    on(event: 'state', listener: (state: TrainingState) => void): void;
    on(event: 'callback', listener: (params: CallbackParameters) => void): void;
    on(event: 'error', listener: (message: string) => void): void;
    on(event: 'info', listener: (message: string) => void): void;

    emit(event: 'state', state: TrainingState): Promise<void>;
    emit(event: 'callback', params: CallbackParameters): Promise<void>;
    emit(event: 'error', message: string): Promise<void>;
    emit(event: 'info', message: string): Promise<void>;
}

/**
 * Interface for controlling the training process.
 */
export interface TrainingControl {
    /**
     * Stops the training process.
     */
    stop(): void;

    /**
     * Pauses the training process.
     */
    pause(): void;

    /**
     * Resumes the training process.
     */
    resume(): void;

    /**
     * Performs a single training step.
     */
    step(): void;

    /**
     * Indicates if the training process has been stopped.
     */
    get isTrainingStopped(): boolean;

    /**
     * Handles control flow for the training process.
     * @param isSyncBackend - Indicates if the backend is synchronous (e.g., CPU). If true, yields control to the event loop.
     */
    handleControlFlow(isSyncBackend?: boolean): Promise<void>;
}

/**
 * Interface for regularization techniques.
 */
export interface Regularization {
    /**
     * Computes the regularization loss.
     *
     * @param theta - The model parameters.
     * @returns Scalar representing the computed regularization loss.
     */
    compute(theta: Tensor2D): Scalar;

    /**
     * Computes the gradient of the regularization loss with respect to the model parameters.
     *
     * @param theta - The model parameters.
     * @returns Tensor2D containing the gradients.
     */
    gradient(theta: Tensor2D): Tensor2D;

    /**
     * Disposes of any resources used by the regularization term.
     */
    dispose?(): void;
}

/**
 * Interface for loss functions.
 */
export interface LossFunction {
    /**
     * Computes the loss between true values and predicted values.
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the computed loss.
     */
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar;

    /**
     * Computes the gradient of the loss function with respect to the model parameters.
     * Used in gradient descent optimization.
     *
     * @param xTrue - The feature matrix (shape: [n_samples, n_features]).
     * @param yTrue - The true values (labels) (shape: [n_samples, 1]).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients.
     */
    parameterGradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D;

    /**
     * Computes the gradient of the loss function with respect to the predictions.
     * Used in backpropagation to update the model parameters.
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Tensor2D containing the gradients with respect to predictions.
     */
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D;

    /**
     * Checks if the loss function uses logits.
     *
     * @returns boolean indicating whether the loss function uses logits.
     */
    usesLogits?(): boolean;

    /**
     * Disposes of any resources used by the loss function.
     */
    dispose?(): void;
}

export interface CriterionFunction {
    /**
     * Computes the impurity score for a set of values.
     *
     * @param yValues - The values to compute the impurity score for.
     * @returns Scalar representing the impurity score.
     */
    impurity(yTrue: number[][]): number;

    /**
     * Computes the loss between true values and predicted values.
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the computed loss.
     */
    loss(yTrue: Tensor2D, yPred: Tensor2D): Scalar;

    /**
     * Disposes of any resources used by the loss function.
     */
    dispose?(): void;
}

export type OptimizerLossFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D) => Scalar;
export type OptimizerGradientFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D) => Tensor2D;

/**
 * Parameters for the optimization process.
 */
export type OptimizeParameters = Readonly<{
    X: Tensor2D;
    y: Tensor2D;
    lossFunction: OptimizerLossFunction;
    gradientFunction: OptimizerGradientFunction;
    initTheta: Tensor2D;
    threadId?: number;
    threadName?: string;
}>;

/**
 * Interface for optimizers.
 */
export interface Optimizer {
    /**
     * Optimizes the model parameters.
     *
     * @param params - The optimization parameters.
     * @returns The optimized model parameters.
     */
    optimize(params: OptimizeParameters): Promise<Tensor2D>;

    /**
     * Disposes of any resources used by the optimizer.
     */
    dispose?(): void;
}

export type TreeNode = {
    leftChild: TreeNode | null;
    rightChild: TreeNode | null;
    readonly featureIndex: number | null;
    readonly threshold: number | null;
    readonly value: number;
    readonly probabilities?: number[]; // Optional for classification tasks
};

export type EnsembleTree = ReadonlyArray<TreeNode>;

/**
 * Parameters for trained Gaussian Naive Bayes model.
 */
export type GaussianNaiveBayesParams = Readonly<{
    type: 'gaussian';
    classes: number[];
    classMeans: MatrixLike;
    classVariances: MatrixLike;
    classPriors: Float32Array;
}>;

/**
 * Parameters for trained Quadratic Naive Bayes model.
 */
export type QuadraticNaiveBayesParams = Readonly<{
    type: 'quadratic';
    classes: number[];
    classMeans: MatrixLike;
    classCovariances: MatrixLike[];
    classCovariancesInverse: MatrixLike[];
    classCovariancesDeterminant: Float32Array;
    classPriors: Float32Array;
}>;

export type NaiveBayesParams = GaussianNaiveBayesParams | QuadraticNaiveBayesParams;

export type ModelRepresentation = Tensor2D | EnsembleTree | NaiveBayesParams;

/**
 * Interface for machine learning models.
 */
export interface Model<T extends ModelRepresentation> {
    /**
     * Trains the model on the provided data.
     *
     * @param X - The input features.
     * @param y - The target labels.
     * @returns A promise that resolves when training is complete.
     */
    train(X: Tensor2D, y: Tensor2D): Promise<T>;

    /**
     * Makes predictions using the trained model.
     *
     * @param X - The input features.
     * @param theta - The model parameters (optional).
     * @returns The predicted values.
     */
    predict(X: Tensor2D, theta?: T): Tensor2D;

    /**
     * Evaluates the model on the provided data.
     *
     * @param X - The input features.
     * @param y - The true labels.
     * @param theta - The model parameters (optional).
     * @returns A tuple containing the predicted values, true labels, and the loss.
     */
    evaluate(X: Tensor2D, y: Tensor2D, theta?: T): [Tensor2D, Tensor2D, Scalar];

    /**
     * Disposes of any resources used by the model.
     *
     * @param withDependencies - Whether to dispose of dependent resources.
     */
    dispose(withDependencies?: boolean): void;

    /**
     * Checks if the model uses one-hot encoding for labels.
     *
     * @returns boolean indicating whether the model uses one-hot encoding.
     */
    usesOneHotLabels?(): boolean;
}
