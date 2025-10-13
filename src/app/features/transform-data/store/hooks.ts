import { useTransformationStore } from './store';

export const useTransformations = () => useTransformationStore((state) => state.transformations);
export const useNormalization = () => useTransformationStore((state) => state.normalization);
