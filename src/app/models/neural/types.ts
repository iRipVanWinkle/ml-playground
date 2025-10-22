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

export type NeuralClassificationTrainingReport = {
    type: 'neural';
    taskType: 'classification';
    trainLossHistory: number[][];
    iteration: number;
    testAccuracy: number;
    trainAccuracy: number;
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    predictionPredictedLabels?: number[][];
    theta: number[][];
};

export type NeuralRegressionTrainingReport = {
    type: 'neural';
    taskType: 'regression';
    trainLossHistory: number[][];
    iteration: number;
    trainLoss: number;
    testLoss: number;
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    predictionPredictedLabels?: number[][];
    theta: number[][];
};
