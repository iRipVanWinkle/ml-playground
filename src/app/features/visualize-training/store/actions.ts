import type { ModelType, TrainingReport } from '@/app/models/types';
import { useVisualizeTrainingStore } from './store';
import { getModelRegistry } from '@/app/models/ui-registry';
import type { TaskType } from '@/app/shared/types';

const registry = getModelRegistry();

export const resetTrainingReport = (modelType?: ModelType, taskType?: TaskType) => {
    const { trainingReport } = useVisualizeTrainingStore.getState();

    modelType = modelType ?? trainingReport.type;
    taskType = taskType ?? trainingReport.taskType;

    const modelDefinition = registry.get(modelType);

    const defaultReport = modelDefinition.defaultReport(taskType);

    useVisualizeTrainingStore.setState({ trainingReport: defaultReport }, true);
};

export const setTrainingReport = (report: TrainingReport) => {
    useVisualizeTrainingStore.setState({ trainingReport: report });
};
