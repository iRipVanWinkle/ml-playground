import type { Tensor2D } from '@tensorflow/tfjs';

export * from './euclidean';
export * from './manhattan';
export * from './cosine';

export type DistanceMetric = (X: Tensor2D, Y: Tensor2D) => Tensor2D;
