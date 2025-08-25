import type { Rank, Scalar, Tensor2D, Variable } from '@tensorflow/tfjs';
import type { EventEmitter } from './events/EventEmitter';

export type Variable2D = Variable<Rank.R2>;

/**
 * Computes the metric between true values and predicted values.
 *
 * @param yTrue - The true values (labels).
 * @param yPred - The predicted values.
 * @returns Scalar representing the computed metric.
 */
export type MetricFunction = (yTrue: Tensor2D, yPred: Tensor2D) => Scalar;

export type OptimizerCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    alfa: number;
    loss: number;
    theta: Tensor2D;
    threadName?: string;
}>;
export type OptimizerCallback = (params: OptimizerCallbackParameters) => Promise<void> | void;

export type TrainingState = 'transforming' | 'training' | 'paused' | 'stopped' | 'stepped-forward';
/**
 * Interface for training event listeners.
 */
export interface TrainingEventEmitter extends EventEmitter {
    on(event: 'state', listener: (state: TrainingState) => void): void;
    on(event: 'callback', listener: (params: OptimizerCallbackParameters) => void): void;
    on(event: 'error', listener: (message: string) => void): void;
    on(event: 'info', listener: (message: string) => void): void;

    emit(event: 'state', state: TrainingState): Promise<void>;
    emit(event: 'callback', params: OptimizerCallbackParameters): Promise<void>;
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
export interface Optimizer extends TrainingControl {
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

/**
 * Interface for machine learning models.
 */
export interface Model<T = unknown> extends TrainingControl {
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
