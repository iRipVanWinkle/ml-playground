import { initState, useTrainingStore } from './store';
import type { TrainingState, PendingAction, TrainingReport } from './types';

export const reset = () => {
    useTrainingStore.setState(initState, true);
};

export const setTrainingStatus = (trainingState: TrainingState) => {
    useTrainingStore.setState({ trainingState });
};

export const setPendingAction = (pendingAction: PendingAction) => {
    useTrainingStore.setState({ pendingAction });
};

export const setTrainingReport = (report: TrainingReport) => {
    useTrainingStore.setState({ trainingReport: report });
};
