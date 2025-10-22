import { create } from 'zustand';
import type { TrainingStore } from './types';

export const initState: TrainingStore = {
    trainingState: 'init',
    pendingAction: null,
};

export const useTrainingStore = create(() => initState);
