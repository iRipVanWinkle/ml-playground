import type { Tensor2D } from '@tensorflow/tfjs';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { BaseClassificationReport } from '@/app/shared/types';
import type { MatrixLike } from '@/app/shared/helpers';

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

export type LogisticTrainingReport = BaseClassificationReport & {
    type: 'logistic';
    trainLossHistory: number[][];
    iterations: number[];
    theta: MatrixLike;
};
