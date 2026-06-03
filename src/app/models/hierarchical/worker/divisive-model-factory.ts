import { DivisiveClustering, AgglomerativeClustering } from '@/ml/models';
import { distanceFactory, arrayClusteringMathFactory } from '@/ml/factories';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { TrainingSettings } from '../../types';
import type { HierarchicalClusteringSettings } from '../types';

export function hierarchicalModelFactory(
    settings: TrainingSettings<HierarchicalClusteringSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { numClusters, distance, method } = settings.modelSettings;
    const distanceMetric = distanceFactory(distance);

    if (method === 'agglomerative') {
        const { linkage } = settings.modelSettings;
        if (linkage === 'ward' && distance.type !== 'euclidean') {
            throw new Error("Ward's linkage method requires the Euclidean distance metric.");
        }
        return new AgglomerativeClustering({
            numClusters,
            linkage,
            distanceMetric,
            eventEmitter,
            trainingController,
        });
    }

    const { bisectIterations, bisectRestarts } = settings.modelSettings;
    const clusteringMath = arrayClusteringMathFactory(distance);

    return new DivisiveClustering({
        numClusters,
        bisectIterations,
        bisectRestarts,
        eventEmitter,
        trainingController,
        distanceFunction: clusteringMath.distance,
        centroidFunction: clusteringMath.centroid,
    });
}
