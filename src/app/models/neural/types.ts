import type { Tensor2D } from '@tensorflow/tfjs';
import type { OptimizerCallbackParameters } from '@/ml/types';
import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { MatrixLike } from '@/app/shared/helpers';
import type { BaseClassificationReport, BaseRegressionReport } from '@/app/shared/types';

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

export type NeuralClassificationTrainingReport = BaseClassificationReport & {
    type: 'neural';
    trainLossHistory: number[][];
    iteration: number;
    theta: MatrixLike;
};

export type NeuralRegressionTrainingReport = BaseRegressionReport & {
    type: 'neural';
    trainLossHistory: number[][];
    iteration: number;
    optimizerLoss: number;
    theta: MatrixLike;
};
