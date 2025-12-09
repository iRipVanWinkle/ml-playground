import type { RegressionMetricsData } from '@/app/shared/visualization';
import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { MatrixLike } from '@/app/shared/helpers';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type { Tensor2D } from '@tensorflow/tfjs';

export type LinearSettings = {
    type: 'linear';
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
};

export type LinearRepresentation = {
    type: 'linear';
    representation: Tensor2D;
};

export type LinearCallbackParameters = {
    type: 'linear';
    callbackParameters: OptimizerCallbackParameters;
};

export type LinearTrainingReport = {
    type: 'linear';
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
