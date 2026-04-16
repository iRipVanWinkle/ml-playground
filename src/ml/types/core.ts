import type { Scalar, Tensor2D, Tensor3D } from '@tensorflow/tfjs';
import type { ModelRepresentation, PredictionMetadata, ScalerParams } from './data';

/**
 * A metric computes a scalar score from predictions vs ground truth.
 * Metrics are used for evaluation only — they have no gradients
 * and are not involved in training.
 */
export type MetricFunction = (yTrue: Tensor2D, yPred: Tensor2D) => Scalar;

/**
 * Aggregates predictions from multiple trees in an ensemble model.
 * For classification, this might be majority voting; for regression,
 * averaging the individual tree predictions.
 */
export type EnsembleAggregatorFn = (predictions: Tensor2D | Tensor3D) => Tensor2D;

/**
 * A loss function measures how far predictions are from ground truth
 * AND provides gradients for optimization. Unlike a metric, a loss
 * function is differentiable and used inside the training loop.
 *
 * - compute(): delegates to the corresponding metric for the scalar value
 * - parameterGradient(): gradient w.r.t. model weights (for gradient descent)
 * - predictionGradient(): gradient w.r.t. predictions (for backpropagation)
 */
export interface LossFunction {
    compute(yTrue: Tensor2D, yPred: Tensor2D): Scalar;
    parameterGradient(xTrue: Tensor2D, yTrue: Tensor2D, yPred: Tensor2D): Tensor2D;
    predictionGradient(yTrue: Tensor2D, yPred: Tensor2D): Tensor2D;
    usesLogits?(): boolean;
    dispose?(): void;
}

/**
 * A criterion wraps a loss function for use in decision tree node splitting.
 * It adds impurity() — a measure of how mixed a node's samples are.
 * Lower impurity means a purer (better) split. Examples: Gini impurity,
 * entropy (classification), or variance/MSE (regression).
 */
export interface CriterionFunction {
    impurity(yTrue: number[][]): number;
    loss(yTrue: Tensor2D, yPred: Tensor2D): Scalar;
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
 * An optimizer updates model parameters to minimize a loss function.
 * It receives the current parameters, computes gradients via the loss
 * function, and returns updated parameters. Different optimizers use
 * different update rules (e.g., vanilla gradient descent, Adam, momentum).
 */
export interface Optimizer {
    optimize(params: OptimizeParameters): Promise<Tensor2D>;
    dispose?(): void;
}

/**
 * Regularization adds a penalty term to the loss function to prevent
 * overfitting. L1 encourages sparsity (some weights become zero),
 * L2 encourages small weights (weight decay), and ElasticNet combines both.
 */
export interface Regularization {
    compute(theta: Tensor2D): Scalar;
    gradient(theta: Tensor2D): Tensor2D;
    dispose?(): void;
}

/**
 * A scaler normalizes input data before training and denormalizes
 * predictions after inference. Normalization improves training stability
 * by ensuring features operate on similar scales.
 */
export interface Scaler<T extends ScalerParams> {
    fit(tensor: Tensor2D): void;
    transform(tensor: Tensor2D): Tensor2D;
    extractParameters?(): Promise<T>;
    restoreParameters?(params: T): void;
    dispose?(): void;
}

/**
 * A model encapsulates the full train → predict lifecycle.
 * The type parameter T represents the model's internal state
 * (e.g., Tensor2D for linear models, EnsembleTree for forests).
 */
export interface Model<T extends ModelRepresentation> {
    train(X: Tensor2D, y: Tensor2D): Promise<T>;
    predict(X: Tensor2D, theta?: T): Tensor2D;
    predictWithMetadata(X: Tensor2D, theta?: T): PredictionMetadata;
    dispose(withDependencies?: boolean): void;
    usesOneHotLabels?(): boolean;
}
