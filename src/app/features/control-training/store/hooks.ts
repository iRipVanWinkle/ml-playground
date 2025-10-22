import { resetTrainingControls } from './actions';
import { useTrainingStore } from './store';

export const useIsTraining = () =>
    useTrainingStore(
        (state) => state.trainingState === 'training' || state.trainingState === 'paused',
    );

export const useTrainingState = () => useTrainingStore((state) => state.trainingState);
export const usePendingAction = () => useTrainingStore((state) => state.pendingAction);
export const useResetTrainingControls = () => resetTrainingControls;
