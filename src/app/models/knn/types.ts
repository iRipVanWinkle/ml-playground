import type { BaseClassificationReport, BaseRegressionReport } from '@/app/shared/types';
import type { KNNCallbackParameters as KNNCallbackParametersType, KNNParams } from '@/ml/types';
import type { DistanceConfig } from '@/ml/factories';

export type KNNWeights = 'uniform' | 'distance';

export type KNNSettings = {
    type: 'knn';
    k: number;
    weights: KNNWeights;
    distance: DistanceConfig;
};

export type KNNRepresentation = {
    type: 'knn';
    representation: KNNParams;
};

export type KNNCallbackParameters = {
    type: 'knn';
    callbackParameters: KNNCallbackParametersType;
};

export type KNNClassificationTrainingReport = BaseClassificationReport & {
    type: 'knn';
    params?: KNNParams;
};

export type KNNRegressionTrainingReport = BaseRegressionReport & {
    type: 'knn';
    params?: KNNParams;
};
