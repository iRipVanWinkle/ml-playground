import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';
import type { TrainingState } from '@/app/shared/types';

type TrainingControlSlice = Pick<
    AppStore,
    'training' | 'setTrainingState' | 'resetTrainingControls'
>;

export const createTrainingControlSlice: StateCreator<AppStore, [], [], TrainingControlSlice> = (
    set,
) => ({
    training: {
        state: 'idle',
    },

    setTrainingState: (state) => set((prev) => ({ training: { ...prev.training, state } })),

    resetTrainingControls: (initState?: TrainingState) => ({
        training: { state: initState ?? 'init' },
    }),
});
