import type { WorkerDefinition } from '@/app/shared/registry';
import { linearModelFactory } from './worker/linear-model-factory';
import { LinearLiveMetrics } from './worker/linear-live-metrics';

export const linearWorkerDefinition: WorkerDefinition<'linear'> = {
    key: 'linear',
    modelFactory: linearModelFactory,
    liveMetricsFactory: LinearLiveMetrics.factory,
};
