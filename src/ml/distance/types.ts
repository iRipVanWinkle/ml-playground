import type { Tensor2D } from '@tensorflow/tfjs';

export type DistanceMetric = (A: Tensor2D, B: Tensor2D) => Tensor2D;

export type DistanceFunction = (a: number[], b: number[]) => number;

export type CentroidFunction = (pts: number[][]) => number[];

export interface ArrayClusteringMath {
    distance: DistanceFunction;
    centroid: CentroidFunction;
}
