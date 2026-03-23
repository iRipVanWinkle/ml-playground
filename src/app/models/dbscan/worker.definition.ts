import type { WorkerDefinition } from '@/app/shared/registry';
import { dbscanModelFactory } from './worker/dbscan-model-factory';
import { DBSCANAnomalyLiveMetrics } from './worker/dbscan-anomaly-live-metrics';
import { DBSCANClusteringLiveMetrics } from './worker/dbscan-clustering-live-metrics';

export const dbscanWorkerDefinition: WorkerDefinition<'dbscan'> = {
    key: 'dbscan',
    modelFactory: dbscanModelFactory,
    liveMetricsFactory: (model, datasetManager, settings) => {
        if (settings.taskType === 'clustering') {
            return DBSCANClusteringLiveMetrics.factory(model, datasetManager, settings);
        } else {
            return DBSCANAnomalyLiveMetrics.factory(model, datasetManager);
        }
    },
    extractParameters: (report) => report.params ?? null,
};
