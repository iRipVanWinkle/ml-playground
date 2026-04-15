import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';

type SystemSlice = Pick<AppStore, 'system' | 'setBackend' | 'setRandomSeed'>;

export const createSystemSlice: StateCreator<AppStore, [], [], SystemSlice> = (set) => ({
    system: {
        backend: 'auto',
        randomSeed: 42,
    },

    setBackend: (backend) => set((state) => ({ system: { ...state.system, backend } })),

    setRandomSeed: (randomSeed) => set((state) => ({ system: { ...state.system, randomSeed } })),
});
