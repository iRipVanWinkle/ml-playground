import { useDataset } from './store';

export const useData = () => useDataset((data) => data);
export const useHasData = () => useDataset((data) => data.trainInputFeatures.length > 0);
export const useNumCategories = () => useDataset((data) => data.categories?.length);
export const useNumTrainInputFeatures = () =>
    useDataset((data) => data.trainInputFeatures[0]?.length ?? 0);
