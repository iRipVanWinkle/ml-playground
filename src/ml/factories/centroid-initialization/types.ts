import type { Tensor2D } from '@tensorflow/tfjs';

export type CentroidInitializationType = 'random' | 'kmeans++' | 'custom';

export type CentroidInitializer = (X: Tensor2D, numClusters: number) => Tensor2D;

type CentroidInitializationBaseConfig = {
    type: 'random' | 'kmeans++';
};

type CentroidInitializationCustomConfig = {
    type: 'custom';
    centroids: Tensor2D;
};

export type CentroidInitializationConfig =
    | CentroidInitializationBaseConfig
    | CentroidInitializationCustomConfig;
