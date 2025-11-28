import type { Tensor2D } from '@tensorflow/tfjs';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type { MatrixLike } from '@/ml/matrix';
import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { ConfusionMatrixData, RocCurveData } from '@/app/shared/visualization';

export type ClassificationType = 'binary' | 'softmax' | 'ovr';

export type LogisticSettings = {
    type: 'logistic';
    classificationType: ClassificationType;
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
};

export type LogisticRepresentation = {
    type: 'logistic';
    representation: Tensor2D;
};

export type LogisticCallbackParameters = {
    type: 'logistic';
    callbackParameters: OptimizerCallbackParameters;
};

export type LogisticTrainingReport = {
    type: 'logistic';
    taskType: 'classification';
    trainLossHistory: number[][];
    iterations: number[];
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
