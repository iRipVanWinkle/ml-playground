import { tensor2d } from '@tensorflow/tfjs';
import type { WorkerDefinition } from '@/app/shared/registry';
import { kMeansModelFactory } from './worker/k-means-model-factory';
import { KMeansLiveMetrics } from './worker/k-means-live-metrics';

export const kMeansWorkerDefinition: WorkerDefinition<'k-means'> = {
    key: 'k-means',
    modelFactory: kMeansModelFactory,
    liveMetricsFactory: KMeansLiveMetrics.factory,

    extractParameters: (report) => {
        const { centroids } = report;

        if (centroids.array.length === 0) {
            return null;
        }

        return tensor2d(centroids.array, centroids.shape);
    },
};
