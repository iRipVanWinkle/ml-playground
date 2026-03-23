import type { WorkerDefinition } from '@/app/shared/registry';
import { isolationForestModelFactory } from './worker/isolation-forest-model-factory';
import { IsolationForestLiveMetrics } from './worker/isolation-forest-live-metrics';

export const isolationForestWorkerDefinition: WorkerDefinition<'isolation-forest'> = {
    key: 'isolation-forest',
    modelFactory: isolationForestModelFactory,
    liveMetricsFactory: IsolationForestLiveMetrics.factory,

    extractParameters: (report) =>
        report.params
            ? {
                  trees: report.params,
                  scoreThreshold: report.scoreThreshold,
              }
            : null,
};
