import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';

type TrainingControlSlice = Pick<
    AppStore,
    'training' | 'setTrainingState' | 'resetTrainingControls'
>;

export const createTrainingControlSlice: StateCreator<AppStore, [], [], TrainingControlSlice> = (
    set,
) => ({
    training: {
        state: 'init',
    },

    setTrainingState: (state) => set((prev) => ({ training: { ...prev.training, state } })),

    resetTrainingControls: () => ({ training: { state: 'init' } }),
});
