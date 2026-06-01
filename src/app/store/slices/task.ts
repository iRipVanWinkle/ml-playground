import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';
import { getModelRegistry } from '@/app/models/ui-registry';

const registry = getModelRegistry();

type TaskSlice = Pick<AppStore, 'taskType' | 'switchTask'>;

export const createTaskSlice: StateCreator<AppStore, [], [], TaskSlice> = (set, get) => ({
    taskType: 'regression',

    switchTask: (taskType) => {
        const state = get();
        const models = registry.getForTask(taskType);
        const modelType = models[0].key;

        set({
            taskType,
            dataset: EMPTY_DATASET,

            ...state.resetModelSettings(modelType, taskType),
            ...state.resetTrainingReport(modelType, taskType),
            ...state.resetTrainingControls('idle'),
            ...state.resetTransformations(),
        });
    },
});

const EMPTY_DATASET = {
    id: null,
    trainInputFeatures: [],
    trainTargetLabels: [],
    testInputFeatures: [],
    testTargetLabels: [],
    xMin: [],
    xMax: [],
    headers: [],
    categories: undefined,
    predictionInputFeatures: undefined,
};
