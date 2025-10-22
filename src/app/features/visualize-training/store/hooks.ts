import { resetTrainingReport } from './actions';
import { useVisualizeTrainingStore } from './store';

export const useTrainingReport = () => useVisualizeTrainingStore((state) => state.trainingReport);

export const useResetTrainingReport = () => resetTrainingReport;
