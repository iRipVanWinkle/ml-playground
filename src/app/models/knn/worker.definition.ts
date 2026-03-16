import type { WorkerDefinition } from '@/app/shared/registry';
import { knnModelFactory } from './worker/knn-model-factory';
import { KNNClassificationLiveMetrics } from './worker/knn-classification-metrics';
import { KNNRegressionLiveMetrics } from './worker/knn-regression-metrics';

export const knnWorkerDefinition: WorkerDefinition<'knn'> = {
    key: 'knn',
    modelFactory: knnModelFactory,

    liveMetricsFactory: (model, datasetManager, settings) => {
        if (settings.taskType === 'classification') {
            return KNNClassificationLiveMetrics.factory(model, datasetManager);
        }
        return KNNRegressionLiveMetrics.factory(model, datasetManager, settings);
    },

    extractParameters: (report) => report.params ?? null,
};
