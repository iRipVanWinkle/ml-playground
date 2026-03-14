import type { WorkerDefinition } from '@/app/shared/registry';
import { dbscanModelFactory } from './worker/dbscan-model-factory';
import { DBSCANLiveMetrics } from './worker/dbscan-live-metrics';

export const dbscanWorkerDefinition: WorkerDefinition<'dbscan'> = {
    key: 'dbscan',
    modelFactory: dbscanModelFactory,
    liveMetricsFactory: DBSCANLiveMetrics.factory,
    extractParameters: (report) => report.params ?? null,
};
