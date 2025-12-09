import { create } from 'zustand';
import type { TrainingStore } from './types';
import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';

export const initState: TrainingStore = {
    trainingReport: {
        type: 'linear',
        taskType: 'regression',
        trainLossHistory: [],
        iteration: 0,
        trainLoss: 0,
        testLoss: 0,
        trainPredictedLabels: EMPTY_MATRIX_LIKE,
        testPredictedLabels: EMPTY_MATRIX_LIKE,
        theta: EMPTY_MATRIX_LIKE,
        trainMetrics: null,
        trainResiduals: EMPTY_MATRIX_LIKE,
    },
};

export const useVisualizeTrainingStore = create(() => initState);
