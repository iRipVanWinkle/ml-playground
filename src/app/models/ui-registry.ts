import { ModelRegistry } from '@/app/shared/registry';
import { linearModelDefinition } from './linear/ui.definition';
import { logisticModelDefinition } from './logistic/ui.definition';
import { neuralModelDefinition } from './neural/ui.definition';
import { treeModelDefinition } from './tree/ui.definition';

export const uiRegistry = new ModelRegistry([
    linearModelDefinition,
    logisticModelDefinition,
    neuralModelDefinition,
    treeModelDefinition,
]);

export function getModelRegistry(): ModelRegistry {
    return uiRegistry;
}
