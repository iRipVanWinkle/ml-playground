import { WorkerRegistry } from '@/app/shared/registry';
import { linearWorkerDefinition } from './linear/worker.definition';
import { logisticWorkerDefinition } from './logistic/worker.definition';
import { neuralWorkerDefinition } from './neural/worker.definition';
import { treeWorkerDefinition } from './tree/worker.definition';
import { naiveBayesWorkerDefinition } from './naive-bayes/worker.definition';
import { kMeansWorkerDefinition } from './k-means/worker.definition';
import { knnWorkerDefinition } from './knn/worker.definition';

export const workerRegistry = new WorkerRegistry([
    linearWorkerDefinition,
    logisticWorkerDefinition,
    neuralWorkerDefinition,
    treeWorkerDefinition,
    naiveBayesWorkerDefinition,
    kMeansWorkerDefinition,
    knnWorkerDefinition,
]);

export function getWorkerRegistry(): WorkerRegistry {
    return workerRegistry;
}
