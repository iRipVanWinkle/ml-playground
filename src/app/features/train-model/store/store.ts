import { create } from 'zustand';
import type { TrainingStore } from './types';

export const initState: TrainingStore = {
    trainingState: 'init',
    pendingAction: null,
    trainingReport: {
        trainLossHistory: [],
        testLoss: 0,
        trainAccuracy: 0,
        testAccuracy: 0,
        iterations: [],
        trainPredictedLabels: [],
        testPredictedLabels: [],
        theta: [],
    },
};

export const useTrainingStore = create(() => initState);
