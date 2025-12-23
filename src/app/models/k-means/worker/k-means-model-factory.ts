import type { TrainingSettings } from '../../types';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { KMeansSettings } from '../types';
import { KMeans } from '@/ml/models/k-means/KMeans';
import { centroidInitializationFactory, distanceFactory } from '@/ml/factories';

export function kMeansModelFactory(
    settings: TrainingSettings<KMeansSettings>,
    eventEmitter: TrainingEventEmitter,
    trainingController: TrainingControl,
) {
    const { modelSettings } = settings;
    const { numClusters, maxIterations, tolerance, centroidInitialization, distance } =
        modelSettings;

    const initializeCentroids = centroidInitializationFactory(centroidInitialization);
    const distanceMetric = distanceFactory(distance);

    return new KMeans({
        numClusters,
        maxIterations,
        tolerance,
        initializeCentroids,
        distanceMetric,
        eventEmitter,
        trainingController,
    });
}
