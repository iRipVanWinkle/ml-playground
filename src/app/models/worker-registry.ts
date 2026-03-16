import { WorkerRegistry } from '@/app/shared/registry';
import { linearWorkerDefinition } from './linear/worker.definition';
import { logisticWorkerDefinition } from './logistic/worker.definition';
import { neuralWorkerDefinition } from './neural/worker.definition';
import { treeWorkerDefinition } from './tree/worker.definition';
import { naiveBayesWorkerDefinition } from './naive-bayes/worker.definition';
import { kMeansWorkerDefinition } from './k-means/worker.definition';
import { knnWorkerDefinition } from './knn/worker.definition';
import { gaussianDistributionWorkerDefinition } from './gaussian-distribution/worker.definition';
import { dbscanWorkerDefinition } from './dbscan/worker.definition';
import { isolationForestWorkerDefinition } from './isolation-forest/worker.definition';
import { hierarchicalClusteringWorkerDefinition } from './hierarchical/worker.definition';

export const workerRegistry = new WorkerRegistry([
    linearWorkerDefinition,
    logisticWorkerDefinition,
    neuralWorkerDefinition,
    treeWorkerDefinition,
    naiveBayesWorkerDefinition,
    kMeansWorkerDefinition,
    knnWorkerDefinition,
    gaussianDistributionWorkerDefinition,
    dbscanWorkerDefinition,
    isolationForestWorkerDefinition,
    hierarchicalClusteringWorkerDefinition,
]);

export function getWorkerRegistry(): WorkerRegistry {
    return workerRegistry;
}
