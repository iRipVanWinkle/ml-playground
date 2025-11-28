import type { WorkerDefinition } from '@/app/shared/registry';
import { naiveBayesModelFactory } from './worker/naive-bayes-model-factory';
import { NaiveBayesLiveMetrics } from './worker/naive-bayes-live-metrics';

export const naiveBayesWorkerDefinition: WorkerDefinition<'naive-bayes'> = {
    key: 'naive-bayes',
    modelFactory: naiveBayesModelFactory,
    liveMetricsFactory: NaiveBayesLiveMetrics.factory,
};
