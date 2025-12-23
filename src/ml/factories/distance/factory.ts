import {
    cosineDistance,
    euclideanDistance,
    manhattanDistance,
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
