import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type { Tensor2D } from '@tensorflow/tfjs';

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
