import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';

type UserExampleSlice = Pick<
    AppStore,
    'userExample' | 'setUserExampleInputs' | 'setUserExamplePrediction' | 'resetUserExample'
>;

export const createUserExampleSlice: StateCreator<AppStore, [], [], UserExampleSlice> = (set) => ({
    userExample: {},

    setUserExampleInputs: (inputs) =>
        set((state) => ({ userExample: { ...state.userExample, inputs } })),

    setUserExamplePrediction: (result) =>
        set((state) => ({ userExample: { ...state.userExample, result } })),

    resetUserExample: () => set({ userExample: {} }),
});
