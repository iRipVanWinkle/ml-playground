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

export type IsolationForestCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    ensemble: IsolationEnsembleTree;
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

export type KMeansCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    centroids: Tensor2D;
    assignments: Tensor2D;
    inertia: number;
}>;

export type HierarchicalClusteringCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    assignments: Int32Array;
    numClusters: number;
    params?: HierarchicalClusteringParams;
}>;

export type CallbackParameters =
    | OptimizerCallbackParameters
    | TreeCallbackParameters
    | IsolationForestCallbackParameters
    | NaiveBayesCallbackParameters
    | KMeansCallbackParameters
    | KNNCallbackParameters
    | GaussianDistributionCallbackParameters
    | DBSCANCallbackParameters
    | HierarchicalClusteringCallbackParameters;

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

export type BaseScalerParams = {
    type: 'zscore' | 'minmax' | 'log' | 'none';
};

export type ZScoreScalerParams = BaseScalerParams & {
    type: 'zscore';
    mean: Float32Array;
    std: Float32Array;
};

export type MinMaxScalerParams = BaseScalerParams & {
    type: 'minmax';
    min: Float32Array;
    max: Float32Array;
};

export type LogScalerParams = BaseScalerParams & {
    type: 'log';
};

export type ScalerParams =
    | ZScoreScalerParams
    | MinMaxScalerParams
    | LogScalerParams
    | BaseScalerParams;

export type ScalerState<T = ScalerParams> = {
    preScaler?: T;
    postScaler?: T;
};

/**
 * Interface for scalers.
 */
export interface Scaler<T extends ScalerParams> {
    /**
     * Fits the scaler to the input tensor.
     *
     * @param tensor - The input tensor.
     */
    fit(tensor: Tensor2D): void;

    /**
     * Transforms the input tensor.
     *
     * @param tensor - The input tensor.
     * @returns The transformed tensor.
     */
    transform(tensor: Tensor2D): Tensor2D;

    /**
     * Extracts parameters from GPU memory to CPU memory as typed arrays.
     */
    extractParameters?(): Promise<T>;

    /**
     * Restores parameters from CPU memory back to GPU memory.
     */
    restoreParameters?(params: T): void;

    /**
     * Disposes of any resources used by the scaler.
     */
    dispose?(): void;
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

export type IsolationEnsembleTree = {
    trees: EnsembleTree;
    scoreThreshold: number;
};

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

/**
 * Parameters for trained K-Nearest Neighbors model.
 */
export type KNNParams = Readonly<{
    type: 'knn';
    XTrain: Tensor2D;
    yTrain: Tensor2D;
    classes: number[];
}>;

export type KNNCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    params: KNNParams;
}>;

/**
 * Parameters for diagonal Gaussian Distribution anomaly detection model.
 */
export type DiagonalGaussianDistributionParams = Readonly<{
    type: 'gaussian-distribution';
    covarianceType: 'diagonal';
    featureMeans: Float32Array;
    featureVariances: Float32Array;
}>;

/**
 * Parameters for full-covariance Gaussian Distribution anomaly detection model.
 */
export type FullGaussianDistributionParams = Readonly<{
    type: 'gaussian-distribution';
    covarianceType: 'full';
    featureMeans: Float32Array;
    covarianceMatrix: MatrixLike;
    covarianceInverse: MatrixLike;
    covarianceDeterminant: number;
}>;

/**
 * Union type for Gaussian Distribution parameters (backward compatible).
 */
export type GaussianDistributionParams =
    | DiagonalGaussianDistributionParams
    | FullGaussianDistributionParams;

export type GaussianDistributionCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    params: GaussianDistributionParams;
}>;

/**
 * Parameters for trained Divisive Clustering model.
 */
export type HierarchicalClusteringParams = Readonly<{
    centroids: MatrixLike;
    assignments: Int32Array;
}>;

/**
 * Parameters for trained DBSCAN clustering model.
 */
export type DBSCANParams = Readonly<{
    type: 'dbscan';
    corePoints: MatrixLike;
    coreLabels: Int32Array;
}>;

export type DBSCANCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    assignments: Int32Array;
    numClusters: number;
    activePointIndex?: number;
    epsilon: number;
    params?: DBSCANParams;
}>;

export type ModelRepresentation =
    | Tensor2D
    | EnsembleTree
    | IsolationEnsembleTree
    | NaiveBayesParams
    | KNNParams
    | GaussianDistributionParams
    | DBSCANParams
    | HierarchicalClusteringParams;

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
     * Makes predictions using the trained model and provides additional metadata.
     *
     * @param X - The input features.
     * @param theta - The model parameters (optional).
     * @returns An object containing the predictions and additional metadata.
     */
    predictWithMetadata(X: Tensor2D, theta?: T): PredictionMetadata;

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

export type ClassificationMetadata = {
    type: 'classification';
    predictions: Tensor2D;
    probabilities: Tensor2D;
    dispose(): void;
};

export type RegressionMetadata = {
    type: 'regression';
    predictions: Tensor2D;
    dispose(): void;
};

export type ClusteringMetadata = {
    type: 'clustering';
    assignments: Tensor2D;
    dispose(): void;
};

export type AnomalyDetectionMetadata = {
    type: 'anomaly-detection';
    predictions: Tensor2D;
    probabilities: Tensor2D;
    dispose(): void;
};

export type PredictionMetadata =
    | ClassificationMetadata
    | RegressionMetadata
    | ClusteringMetadata
    | AnomalyDetectionMetadata;
