import { create } from 'zustand';
import type { TrainingStore } from './types';

export const initState: TrainingStore = {
    trainingReport: {
        type: 'linear',
        taskType: 'regression',
        trainLossHistory: [],
        iteration: 0,
        trainLoss: 0,
        testLoss: 0,
        trainPredictedLabels: [],
        testPredictedLabels: [],
        theta: [],
    },
};

export const useVisualizeTrainingStore = create(() => initState);
