import { create } from 'zustand';
import type { AppStore } from './types';
import { createTaskSlice } from './slices/task';
import { createDatasetSlice } from './slices/dataset';
import { createModelSettingsSlice } from './slices/model-settings';
import { createTransformationsSlice } from './slices/transformations';
import { createSystemSlice } from './slices/system';
import { createTrainingControlSlice } from './slices/training-control';
import { createTrainingReportSlice } from './slices/training-report';
import { createUserExampleSlice } from './slices/user-example';

export const useAppStore = create<AppStore>()((...args) => ({
    ...createTaskSlice(...args),
    ...createDatasetSlice(...args),
    ...createModelSettingsSlice(...args),
    ...createTransformationsSlice(...args),
    ...createSystemSlice(...args),
    ...createTrainingControlSlice(...args),
    ...createTrainingReportSlice(...args),
    ...createUserExampleSlice(...args),

    snapshotTrainingSettings: () => {
        const state = args[1]();
        return {
            taskType: state.taskType,
            modelSettings: state.modelSettings,
            systemSettings: state.system,
            dataSettings: state.transformations,
            dataset: state.dataset,
        };
    },
}));
