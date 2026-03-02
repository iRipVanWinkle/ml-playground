import { tensor2d } from '@tensorflow/tfjs';
import type { WorkerDefinition } from '@/app/shared/registry';
import { logisticModelFactory } from './worker/logistic-model-factory';
import { LogisticLiveMetrics } from './worker/logistic-live-metrics';

export const logisticWorkerDefinition: WorkerDefinition<'logistic'> = {
    key: 'logistic',
    modelFactory: logisticModelFactory,
    liveMetricsFactory: LogisticLiveMetrics.factory,

    extractParameters: (report) => {
        const { theta } = report;

        if (theta.array.length === 0) {
            return null;
        }

        return tensor2d(theta.array, theta.shape);
    },
};
