import type { TrainingReport } from '@/app/models/types';
import { initState, useVisualizeTrainingStore } from './store';

export const resetTrainingReport = () => {
    useVisualizeTrainingStore.setState(initState, true);
};

export const setTrainingReport = (report: TrainingReport) => {
    useVisualizeTrainingStore.setState({ trainingReport: report });
};
