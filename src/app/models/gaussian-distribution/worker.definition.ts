import type { WorkerDefinition } from '@/app/shared/registry';
import { gaussianDistributionModelFactory } from './worker/gaussian-distribution-model-factory';
import { GaussianDistributionLiveMetrics } from './worker/gaussian-distribution-live-metrics';

export const gaussianDistributionWorkerDefinition: WorkerDefinition<'gaussian-distribution'> = {
    key: 'gaussian-distribution',
    modelFactory: gaussianDistributionModelFactory,
    liveMetricsFactory: GaussianDistributionLiveMetrics.factory,

    extractParameters: (report) => report.params ?? null,
};
