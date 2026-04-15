import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
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
    trainLoss: 0,
    optimizerLoss: 0,
    trainPredictedLabels: EMPTY_MATRIX_LIKE,
    testPredictedLabels: EMPTY_MATRIX_LIKE,
    theta: EMPTY_MATRIX_LIKE,
    trainMetrics: null,
    trainResiduals: EMPTY_MATRIX_LIKE,
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
