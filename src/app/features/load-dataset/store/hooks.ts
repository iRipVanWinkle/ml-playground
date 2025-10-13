import { useDatasetStore } from './store';

export const useNumTrainInputFeatures = () =>
    useDatasetStore((data) => data.trainInputFeatures[0]?.length ?? 0);
export const useNumCategories = () => useDatasetStore((data) => data.categories?.length ?? 0);
export const useHasData = () => useDatasetStore((data) => data.trainInputFeatures.length > 0);
