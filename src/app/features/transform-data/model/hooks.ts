import { updateNormalization, updateTransformations } from './actions';
import { useTransformationSettings } from './store';

export const useTransformations = () => {
    const transformations = useTransformationSettings((state) => state.transformations);

    return [transformations, updateTransformations] as const;
};

export const useNormalization = () => {
    const randomSeed = useTransformationSettings((state) => state.normalization);

    return [randomSeed, updateNormalization] as const;
};
