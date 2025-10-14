import type { ModelSettings, ModelType } from '@/app/models/types';
import { useModelSettingsStore } from './store';
import type { LossFunctionType, ThetaInitializationConfig } from '@/ml/factories';
import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';

function prefillClassificationSettings(newSettings: Partial<Omit<ModelSettings, 'type'>>) {
    // Only prefill if classificationType is being set and related fields are missing
    if ('classificationType' in newSettings) {
        const classificationType = newSettings.classificationType;
        let lossType: LossFunctionType = 'binaryCrossentropy';
        let initType: ThetaInitializationConfig['type'] = 'zeros';

        if (classificationType === 'softmax') {
            lossType = 'categoricalCrossentropy';
            initType = 'xavierUniform';
        }

        return {
            ...newSettings,
            lossFunction: { type: lossType },
            thetaInitialization: { type: initType },
        };
    }

    return newSettings;
}

export function updateModelSettings(newSettings: Partial<Omit<ModelSettings, 'type'>>) {
    const updatedSettings = prefillClassificationSettings(newSettings);
    useModelSettingsStore.setState(updatedSettings);
}

const registry = getModelRegistry();

export function setModelType(modelType: ModelType, taskType: TaskType) {
    const modelDefinition = registry.get(modelType);
    useModelSettingsStore.setState(modelDefinition.defaultSettings(taskType), true);
}
