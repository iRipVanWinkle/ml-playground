import type { MatrixLike } from './helpers';
import type { ScalerState } from '@/ml/types';
import type { ConfusionMatrixData, RegressionMetricsData, RocCurveData } from './visualization';

export type TaskType =
    | 'regression'
    | 'classification'
    | 'clustering'
    | 'anomaly'
    | 'recommendation';

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

export type TransformationType = 'sinusoid' | 'cosinusoid' | 'fourier' | 'polynomial';

export type Transformation = {
    type: TransformationType;
    degree: number;
};

export type BaseTrainingReport = {
    type: string;
    taskType: TaskType;
    scaler?: ScalerState;
};

export type BaseRegressionReport = BaseTrainingReport & {
    taskType: 'regression';
    trainLoss: number;
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
    trainAccuracy: number;
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
