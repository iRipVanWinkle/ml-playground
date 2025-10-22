import { initState, useTrainingStore } from './store';
import type { TrainingState, PendingAction } from './types';

export const resetTrainingControls = () => {
    useTrainingStore.setState(initState, true);
};

export const setTrainingStatus = (trainingState: TrainingState) => {
    useTrainingStore.setState({ trainingState });
};

export const setPendingAction = (pendingAction: PendingAction) => {
    useTrainingStore.setState({ pendingAction });
};
