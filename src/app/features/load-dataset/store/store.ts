import { create } from 'zustand';
import type { DataState } from './types';

export const initState: DataState = {
    dataset: {
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
};

export const useDatasetStore = create(() => initState);
