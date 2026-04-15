import type { StateCreator } from 'zustand';
import type { AppStore } from '../types';

type TransformationsSlice = Pick<
    AppStore,
    'transformations' | 'setTransformations' | 'setNormalization' | 'resetTransformations'
>;

const INIT_TRANSFORMATIONS = {
    normalization: 'none' as const,
    transformations: [] as const,
};

export const createTransformationsSlice: StateCreator<AppStore, [], [], TransformationsSlice> = (
    set,
) => ({
    transformations: { ...INIT_TRANSFORMATIONS, transformations: [] },

    setTransformations: (transformations) =>
        set((state) => ({
            transformations: { ...state.transformations, transformations },
        })),

    setNormalization: (normalization) =>
        set((state) => ({
            transformations: { ...state.transformations, normalization },
        })),

    resetTransformations: () => ({
        transformations: { transformations: [], normalization: 'none' },
    }),
});
