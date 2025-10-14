import type { WorkerDefinition } from '@/app/shared/registry';
import { logisticModelFactory } from './worker/logistic-model-factory';

export const logisticWorkerDefinition: WorkerDefinition<'logistic'> = {
    key: 'logistic',
    modelFactory: logisticModelFactory,
};
