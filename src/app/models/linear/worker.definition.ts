import type { WorkerDefinition } from '@/app/shared/registry';
import { linearModelFactory } from './worker/linear-model-factory';

export const linearWorkerDefinition: WorkerDefinition<'linear'> = {
    key: 'linear',
    modelFactory: linearModelFactory,
};
