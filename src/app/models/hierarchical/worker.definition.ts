import type { WorkerDefinition } from '@/app/shared/registry';
import { hierarchicalModelFactory } from './worker/divisive-model-factory';
import { HierarchicalLiveMetrics } from './worker/hierarchical-live-metrics';

export const hierarchicalClusteringWorkerDefinition: WorkerDefinition<'hierarchical'> = {
    key: 'hierarchical',
    modelFactory: hierarchicalModelFactory,
    liveMetricsFactory: HierarchicalLiveMetrics.factory,
    extractParameters: (report) => report.params ?? null,
};
