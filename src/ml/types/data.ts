import type { Rank, Tensor2D, Variable } from '@tensorflow/tfjs';
import type { MatrixLike } from '../utils/matrix';

export type Variable2D = Variable<Rank.R2>;

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

export type ModelRepresentation =
    | Tensor2D
    | EnsembleTree
    | IsolationEnsembleTree
    | NaiveBayesParams
    | KNNParams
    | GaussianDistributionParams
    | DBSCANParams
    | HierarchicalClusteringParams;

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
