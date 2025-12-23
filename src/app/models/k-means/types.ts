import type { MatrixLike } from '@/app/shared/helpers';
import type { KMeansMetricsData } from '@/app/shared/visualization';
import type { CentroidInitializationConfig, DistanceConfig } from '@/ml/factories';
import type { KMeansCallbackParameters as KMeansCallbackParametersMl } from '@/ml/types';
import type { Tensor2D } from '@tensorflow/tfjs';

export type KMeansSettings = {
    type: 'k-means';
    numClusters: number;
    maxIterations: number;
    tolerance?: number;
    centroidInitialization: CentroidInitializationConfig;
    distance: DistanceConfig;
};

export type KMeansRepresentation = {
    type: 'k-means';
    representation: Tensor2D;
};

export type KMeansCallbackParameters = {
    type: 'k-means';
    callbackParameters: KMeansCallbackParametersMl;
};

export type KMeansTrainingReport = {
    type: 'k-means';
    taskType: 'clustering';
    iteration: number;
    centroids: MatrixLike;
    trainAssignments: MatrixLike;
    testAssignments?: MatrixLike;
    inertiaHistory: number[];
    trainMetrics?: KMeansMetricsData;
    testMetrics?: KMeansMetricsData;
};
