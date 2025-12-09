import { create } from 'zustand';
import type { TransformationSettings } from './types';

export const initState: TransformationSettings = {
    normalization: 'none',
    transformations: [],
};

export const useTransformationStore = create(() => initState);
export const useTransformations = () => useTransformationStore((state) => state.transformations);
