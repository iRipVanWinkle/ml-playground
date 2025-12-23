import { euclideanDistance } from '../../distance';
import {
    customCentroidInitFactory,
    kmeansPlusPlusCentroidInitFactory,
    randomCentroidInit,
} from './initializers';
import type { CentroidInitializationConfig, CentroidInitializer } from './types';

export function centroidInitializationFactory(
    centroidInitialization: CentroidInitializationConfig,
): CentroidInitializer {
    switch (centroidInitialization.type) {
        case 'kmeans++':
            return kmeansPlusPlusCentroidInitFactory(euclideanDistance);
        case 'custom':
            return customCentroidInitFactory(centroidInitialization.centroids);
        case 'random':
        default:
            return randomCentroidInit;
    }
}
