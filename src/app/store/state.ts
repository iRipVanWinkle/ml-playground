import { create } from 'zustand';
import type { State } from './types';

export const initState: State = {
    taskType: 'regression',
    dataSettings: {
        normalization: 'none',
        transformations: [],
    },
    modelSettings: {
        type: 'linear',
        lossFunction: { type: 'mse' },
        optimizer: {
            type: 'batch',
            maxIterations: 100,
            tolerance: 0.0001,
            learningRate: 0.01,
            scheduler: false,
            schedulerConfig: { s0: 1, p: 0.5 },
        },
        regularization: {
            type: 'none',
        },
    },
    data: {
        trainInputFeatures: [],
        trainTargetLabels: [],
        testInputFeatures: [],
        testTargetLabels: [],
        xMin: [],
        xMax: [],
        headers: [],
        categories: undefined,
        predictionInputFeatures: undefined,
    },
    trainingState: 'init',
    report: {
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

export const useAppState = create(() => initState);
