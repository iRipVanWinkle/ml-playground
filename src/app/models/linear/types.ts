import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { MatrixLike } from '@/app/shared/helpers';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type { Tensor2D } from '@tensorflow/tfjs';
import type { BaseRegressionReport } from '@/app/shared/types';

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

export type LinearTrainingReport = BaseRegressionReport & {
    type: 'linear';
    trainLossHistory: number[][];
    iteration: number;
    optimizerLoss: number; // optimizer loss
    theta: MatrixLike;
};
