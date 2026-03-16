import { ModelRegistry, type ModelDefinition } from '@/app/shared/registry';
import { linearModelDefinition } from './linear/ui.definition';
import { logisticModelDefinition } from './logistic/ui.definition';
import { neuralModelDefinition } from './neural/ui.definition';
import { treeModelDefinition } from './tree/ui.definition';
import { naiveBayesModelDefinition } from './naive-bayes/ui.definition';
import type { ModelType } from './types';
import { kMeansModelDefinition } from './k-means/ui.definition';
import { knnModelDefinition } from './knn/ui.definition';
import { gaussianDistributionModelDefinition } from './gaussian-distribution/ui.definition';
import { dbscanModelDefinition } from './dbscan/ui.definition';
import { isolationForestModelDefinition } from './isolation-forest/ui.definition';
import { hierarchicalClusteringModelDefinition } from './hierarchical/ui.definition';

export const uiRegistry = new ModelRegistry([
    linearModelDefinition,
    logisticModelDefinition,
    neuralModelDefinition,
    treeModelDefinition,
    naiveBayesModelDefinition,
    kMeansModelDefinition,
    knnModelDefinition,
    gaussianDistributionModelDefinition,
    isolationForestModelDefinition,
    dbscanModelDefinition,
    hierarchicalClusteringModelDefinition,
]);

export function getModelRegistry(): ModelRegistry {
    return uiRegistry;
}

export function useModelDefinition(modelType: ModelType): ModelDefinition {
    return uiRegistry.get(modelType);
}
