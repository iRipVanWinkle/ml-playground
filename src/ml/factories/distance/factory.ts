import {
    cosineDistance,
    euclideanDistance,
    manhattanDistance,
    EuclideanClusteringMath,
    ManhattanClusteringMath,
    CosineClusteringMath,
    type ArrayClusteringMath,
    type DistanceMetric,
} from '../../distance';
import type { DistanceConfig } from './types';

export function distanceFactory(distanceConfig: DistanceConfig): DistanceMetric {
    switch (distanceConfig.type) {
        case 'manhattan':
            return manhattanDistance;
        case 'cosine':
            return cosineDistance;
        case 'euclidean':
        default:
            return euclideanDistance;
    }
}

export function arrayClusteringMathFactory(distanceConfig: DistanceConfig): ArrayClusteringMath {
    switch (distanceConfig.type) {
        case 'manhattan':
            return new ManhattanClusteringMath();
        case 'cosine':
            return new CosineClusteringMath();
        case 'euclidean':
        default:
            return new EuclideanClusteringMath();
    }
}
