import { useTransformationSettings } from './store';

export const useTransformations = () => useTransformationSettings((state) => state.transformations);

export const useNormalization = () => useTransformationSettings((state) => state.normalization);
