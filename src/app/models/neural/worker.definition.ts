import type { WorkerDefinition } from '@/app/shared/registry';
import { neuralModelFactory } from './worker/neural-model-factory';

export const neuralWorkerDefinition: WorkerDefinition<'neural'> = {
    key: 'neural',
    modelFactory: neuralModelFactory,
};
