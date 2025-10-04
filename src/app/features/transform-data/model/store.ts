import { create } from 'zustand';
import type { TransformationSettings } from './types';

const initState: TransformationSettings = {
    normalization: 'none',
    transformations: [],
};

export const useTransformationSettings = create(() => initState);
