import { ModelRegistry, type ModelDefinition } from '@/app/shared/registry';
import { linearModelDefinition } from './linear/ui.definition';
import { logisticModelDefinition } from './logistic/ui.definition';
import { neuralModelDefinition } from './neural/ui.definition';
import { treeModelDefinition } from './tree/ui.definition';
import type { ModelType } from './types';

export const uiRegistry = new ModelRegistry([
    linearModelDefinition,
    logisticModelDefinition,
    neuralModelDefinition,
    treeModelDefinition,
]);

export function getModelRegistry(): ModelRegistry {
    return uiRegistry;
}

export function useModelDefinition(modelType: ModelType): ModelDefinition {
    return uiRegistry.get(modelType);
}
