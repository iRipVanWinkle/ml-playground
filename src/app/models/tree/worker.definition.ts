import type { WorkerDefinition } from '@/app/shared/registry';
import { treeModelFactory } from './worker/tree-model-factory';

export const treeWorkerDefinition: WorkerDefinition<'tree'> = {
    key: 'tree',
    modelFactory: treeModelFactory,
};
