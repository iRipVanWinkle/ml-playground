import type {
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';
import type { Tensor2D } from '@tensorflow/tfjs';
import type { OptimizerCallbackParameters } from '@/ml/types';

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
