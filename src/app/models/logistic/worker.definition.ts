import type { WorkerDefinition } from '@/app/shared/registry';
import { logisticModelFactory } from './worker/logistic-model-factory';
import { LogisticLiveMetrics } from './worker/logistic-live-metrics';

export const logisticWorkerDefinition: WorkerDefinition<'logistic'> = {
    key: 'logistic',
    modelFactory: logisticModelFactory,
    liveMetricsFactory: LogisticLiveMetrics.factory,
};
