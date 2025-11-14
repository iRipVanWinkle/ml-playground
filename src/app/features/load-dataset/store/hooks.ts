import { reset } from './actions';
import { useDatasetStore } from './store';

export const useDataset = () => useDatasetStore(({ dataset }) => dataset);
export const useNumTrainInputFeatures = () =>
    useDatasetStore(({ dataset }) => dataset.trainInputFeatures[0]?.length ?? 0);
export const useNumCategories = () =>
    useDatasetStore(({ dataset }) => dataset.categories?.length ?? 0);
export const useHasData = () =>
    useDatasetStore(({ dataset }) => dataset.trainInputFeatures.length > 0);
export const useResetDataset = () => reset;
