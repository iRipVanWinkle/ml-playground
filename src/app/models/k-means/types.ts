import type { MatrixLike } from '@/app/shared/helpers';
import type { KMeansMetricsData } from '@/app/shared/visualization';
import type { CentroidInitializationConfig, DistanceConfig } from '@/ml/factories';
import type { KMeansCallbackParameters as KMeansCallbackParametersMl } from '@/ml/types';
import type { Tensor2D } from '@tensorflow/tfjs';
import type { BaseClusteringReport } from '@/app/shared/types';

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

export type KMeansTrainingReport = BaseClusteringReport & {
    type: 'k-means';
    iteration: number;
    centroids: MatrixLike;
    trainAssignments: MatrixLike;
    testAssignments?: MatrixLike;
    inertiaHistory: number[];
    trainMetrics?: KMeansMetricsData;
    testMetrics?: KMeansMetricsData;
};
