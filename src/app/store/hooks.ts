import { useAppState } from './state';

export const useModelType = () => useAppState((state) => state.modelSettings.type);
export const useTaskType = () => useAppState((state) => state.taskType);

export const useModelSettings = () => useAppState((state) => state.modelSettings);
export const useClassificationType = () =>
    useAppState((state) =>
        state.modelSettings.type === 'logistic'
            ? state.modelSettings.classificationType
            : undefined,
    );

export const useIsTraining = () =>
    useAppState((state) => state.trainingState === 'training' || state.trainingState === 'paused');
export const useIsPaused = () => useAppState((state) => state.trainingState === 'paused');
export const useIsInit = () => useAppState((state) => state.trainingState === 'init');
export const useTrainingState = () => useAppState((state) => state.trainingState);
export const usePendingAction = () => useAppState((state) => state.pendingAction);
export const useTrainingReport = () => useAppState((state) => state.report);
