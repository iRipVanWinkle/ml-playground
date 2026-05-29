import type { MatrixLike } from './helpers';
import type { ScalerState } from '@/ml/types';
import type { ConfusionMatrixData, RegressionMetricsData, RocCurveData } from './visualization';

/**
 * Task Types
 */

export type TaskType =
    | 'regression'
    | 'classification'
    | 'clustering'
    | 'anomaly'
    | 'recommendation';

/**
 * Dataset
 */

export type Dataset = {
    id: string | null;
    trainInputFeatures: number[][];
    trainTargetLabels: number[][];
    testInputFeatures: number[][];
    testTargetLabels: number[][];
    predictionInputFeatures?: number[][];
    xMin: number[];
    xMax: number[];
    headers: string[];
    categories?: string[];
    isImage?: boolean;
};

/**
 * System Settings
 */

export type TensorBackend = 'auto' | 'webgpu' | 'webgl' | 'cpu' | 'wasm';

export type SystemSettings = {
    backend: TensorBackend;
    randomSeed?: number;
};

/**
 * Transformation Settings
 */

export type NormalizationMethod = 'none' | 'zscore' | 'linear' | 'log';

export type TransformationType = 'sinusoid' | 'cosinusoid' | 'fourier' | 'polynomial';

export type Transformation = {
    type: TransformationType;
    degree: number;
};

/** Represents a transformation entry before the type has been selected. */
export type DraftTransformation = {
    type: '';
    degree: number;
};

export type AnyTransformation = Transformation | DraftTransformation;

export type TransformationSettings = {
    normalization: NormalizationMethod;
    transformations: AnyTransformation[];
};

export type UserExample = {
    inputs?: number[];
    result?: {
        prediction: number;
        probabilities?: number[];
    };
};

/**
 * Training Control
 */

export type TrainingState = 'init' | 'preparing' | 'training' | 'paused';

/**
 * Base Training Reports
 */

export type BaseTrainingReport = {
    type: string;
    taskType: TaskType;
    scaler?: ScalerState;
};

export type BaseRegressionReport = BaseTrainingReport & {
    taskType: 'regression';
    trainLoss?: number;
    testLoss?: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;
    trainMetrics: RegressionMetricsData | null;
    testMetrics?: RegressionMetricsData | null;
    trainResiduals: MatrixLike;
    testResiduals?: MatrixLike;
};

export type BaseClassificationReport = BaseTrainingReport & {
    taskType: 'classification';
    trainAccuracy?: number;
    testAccuracy?: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;

    trainConfusionMatrix: ConfusionMatrixData;
    testConfusionMatrix?: ConfusionMatrixData;

    trainRocCurve: RocCurveData;
    testRocCurve?: RocCurveData;
};

export type BaseClusteringReport = BaseTrainingReport & {
    taskType: 'clustering';
};

export type BaseAnomalyReport = BaseTrainingReport & {
    taskType: 'anomaly';
    trainAnomalyRate?: number;
    testAnomalyRate?: number;

    trainPredictions?: MatrixLike;
    testPredictions?: MatrixLike;
};
