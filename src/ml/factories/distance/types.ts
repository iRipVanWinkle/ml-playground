import type { Tensor2D } from '@tensorflow/tfjs';

export type DistanceType = 'euclidean' | 'cosine' | 'manhattan';

export type CentroidInitializer = (X: Tensor2D, numClusters: number) => Tensor2D;

export type DistanceConfig = {
    type: DistanceType;
};
