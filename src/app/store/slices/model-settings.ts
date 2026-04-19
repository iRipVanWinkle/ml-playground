import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';
import type { LossFunctionType, ThetaInitializationConfig } from '@/ml/factories';
import { getModelRegistry } from '@/app/models/ui-registry';
import { linearModelDefinition } from '@/app/models/linear/ui.definition';

const registry = getModelRegistry();

type Actions = 'modelSettings' | 'updateModelSettings' | 'setModelType' | 'resetModelSettings';
type ModelSettingsSlice = Pick<AppStore, Actions>;

export const createModelSettingsSlice: StateCreator<AppStore, [], [], ModelSettingsSlice> = (
    set,
    get,
) => ({
    modelSettings: linearModelDefinition.defaultSettings('regression'),

    updateModelSettings: (patch) => {
        const updatedPatch = prefillClassificationSettings(patch);
        set((state) => ({ modelSettings: { ...state.modelSettings, ...updatedPatch } }));
    },

    resetModelSettings: (modelType, taskType) => {
        const modelDefinition = registry.get(modelType);

        return { modelSettings: modelDefinition.defaultSettings(taskType) };
    },

    setModelType: (modelType) => {
        const state = get();
        const taskType = state.taskType;

        set({
            ...state.resetModelSettings(modelType, taskType),
            ...state.resetTrainingReport(modelType, taskType),
            ...state.resetTrainingControls(),
        });
    },
});

function prefillClassificationSettings(
    newSettings: Record<string, unknown>,
): Record<string, unknown> {
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
