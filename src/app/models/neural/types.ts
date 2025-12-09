import type { Tensor2D } from '@tensorflow/tfjs';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { MatrixLike } from '@/app/shared/helpers';
import type {
    ConfusionMatrixData,
    RegressionMetricsData,
    RocCurveData,
} from '@/app/shared/visualization';

export type NeuralSettings = {
    type: 'neural';
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
    layers: Array<{ units: number; activation?: string }>;
};

export type NeuralRepresentation = {
    type: 'neural';
    representation: Tensor2D;
};

export type NeuralCallbackParameters = {
    type: 'neural';
    callbackParameters: OptimizerCallbackParameters;
};

export type NeuralClassificationTrainingReport = {
    type: 'neural';
    taskType: 'classification';
    trainLossHistory: number[][];
    iteration: number;
    testAccuracy: number;
    trainAccuracy: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;
    theta: MatrixLike;

    trainConfusionMatrix: ConfusionMatrixData;
    testConfusionMatrix?: ConfusionMatrixData;

    trainRocCurve: RocCurveData;
    testRocCurve?: RocCurveData;
};

export type NeuralRegressionTrainingReport = {
    type: 'neural';
    taskType: 'regression';
    trainLossHistory: number[][];
    iteration: number;
    trainLoss: number;
    testLoss: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;
    theta: MatrixLike;
    trainMetrics: RegressionMetricsData | null;
    testMetrics?: RegressionMetricsData | null;
    trainResiduals: MatrixLike;
    testResiduals?: MatrixLike;
};
