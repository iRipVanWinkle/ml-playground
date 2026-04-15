import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';
import type { Dataset } from '@/app/shared/types';

type DatasetSlice = Pick<AppStore, 'dataset' | 'setDataset'>;

const EMPTY_DATASET: Dataset = {
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

export const createDatasetSlice: StateCreator<AppStore, [], [], DatasetSlice> = (set, get) => ({
    dataset: EMPTY_DATASET,

    setDataset: (dataset) => {
        const state = get();
        set({
            dataset,
            ...state.resetTrainingReport(),
            ...state.resetTrainingControls(),
        });
    },
});
