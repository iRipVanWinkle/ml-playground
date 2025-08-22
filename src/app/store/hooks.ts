import { useAppState } from './state';

export const useModelType = () => useAppState((state) => state.modelSettings.type);
export const useTaskType = () => useAppState((state) => state.taskType);

export const useModelSettings = () => useAppState((state) => state.modelSettings);
export const useDataSettings = () => useAppState((state) => state.dataSettings);
export const useClassificationType = () =>
    useAppState((state) => state.modelSettings.classificationType);

export const useData = () => useAppState((state) => state.data);
export const useHasData = () => useAppState((state) => state.data.trainInputFeatures.length > 0);
export const useNumCategories = () => useAppState((state) => state.data.categories?.length);
export const useNumTrainInputFeatures = () =>
    useAppState((state) => state.data.trainInputFeatures[0]?.length ?? 0);

export const useIsTraining = () =>
    useAppState((state) => state.trainingState === 'training' || state.trainingState === 'paused');
export const useIsPaused = () => useAppState((state) => state.trainingState === 'paused');
export const useIsInit = () => useAppState((state) => state.trainingState === 'init');
export const useTrainingState = () => useAppState((state) => state.trainingState);
export const usePendingAction = () => useAppState((state) => state.pendingAction);
export const useTrainingReport = () => useAppState((state) => state.report);
