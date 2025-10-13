import type { ModelSettings, ModelType } from './types';
import { useModelSettingsStore, initState } from './store';
import type { LossFunctionType, ThetaInitializationConfig } from '@/ml/factories';
import type { TaskType } from '@/app/shared/types';
import { modelSettingsDefaults } from '../defaults';

function prefillClassificationSettings(newSettings: Partial<Omit<ModelSettings, 'type'>>) {
    // Only prefill if classificationType is being set and related fields are missing
    console.info('classificationType' in newSettings);
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

export function reset() {
    useModelSettingsStore.setState(initState, true);
}

export function updateModelSettings(newSettings: Partial<Omit<ModelSettings, 'type'>>) {
    const updatedSettings = prefillClassificationSettings(newSettings);
    useModelSettingsStore.setState(updatedSettings);
}

export function setModelType(modelType: ModelType, taskType: TaskType) {
    useModelSettingsStore.setState(modelSettingsDefaults[modelType](taskType), true);
}
