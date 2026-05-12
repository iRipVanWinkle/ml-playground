import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { createEmptyMatrix } from '@/app/shared/helpers';
import type { TrainingReport } from '@/app/models/types';

const registry = getModelRegistry();

type TrainingReportSlice = Pick<
    AppStore,
    'trainingReport' | 'setTrainingReport' | 'resetTrainingReport'
>;

const INIT_REPORT: TrainingReport = {
    type: 'linear',
    taskType: 'regression',
    trainLossHistory: [],
    iteration: 0,
    optimizerLoss: 0,
    trainPredictedLabels: createEmptyMatrix(),
    testPredictedLabels: createEmptyMatrix(),
    theta: createEmptyMatrix(),
    trainMetrics: null,
    trainResiduals: createEmptyMatrix(),
};

export const createTrainingReportSlice: StateCreator<AppStore, [], [], TrainingReportSlice> = (
    set,
    get,
) => ({
    trainingReport: INIT_REPORT,

    setTrainingReport: (report) => set({ trainingReport: report }),

    resetTrainingReport: (modelType?, taskType?) => {
        const state = get();
        const type = modelType ?? state.trainingReport.type;
        const task = taskType ?? state.trainingReport.taskType;

        const modelDefinition = registry.get(type);
        const defaultReport = modelDefinition.defaultReport(task);

        return { trainingReport: defaultReport };
    },
});
