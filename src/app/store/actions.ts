import type {
    ModelType,
    State,
    TaskType,
    TrainingReport,
    TrainingState,
    PendingAction,
    ModelSettings,
} from './types';
import type { ThetaInitializationConfig, LossFunctionType } from '@/ml/factories';
import { initState, useAppState } from './state';
import { modelSettingsDefaults } from './defaults';
import { resetData } from '../features/load-dataset';

export function setTaskType(taskType: TaskType) {
    const modelType = taskType === 'regression' ? 'linear' : 'logistic';
    const modelSettings = modelSettingsDefaults[modelType](taskType);

    useAppState.setState((state) => ({
        ...state,
        taskType,
        modelSettings,
    }));

    resetTrainingReport();
    resetData();
}

export function setModelType(modelType: ModelType) {
    useAppState.setState((state) => {
        const modelSettings = modelSettingsDefaults[modelType](state.taskType);

        return {
            ...state,
            modelSettings,
        };
    });
}

function prefillClassificationSettings(newSettings: Partial<Omit<ModelSettings, 'type'>>) {
    // Only prefill if classificationType is being set and related fields are missing
    if ('classificationType' in newSettings) {
        const classificationType = newSettings.classificationType;
        let lossType: LossFunctionType = 'binaryCrossentropy';
        let initType: ThetaInitializationConfig['type'] = 'zeros';

        if (classificationType === 'softmax') {
            lossType = 'categoricalCrossentropy';
            initType = 'xavierUniform';
        }

        return {
            ...newSettings,
            lossFunction: { type: lossType },
            thetaInitialization: { type: initType },
        };
    }

    return newSettings;
}

export function updateModelSettings(newSettings: Partial<Omit<ModelSettings, 'type'>>) {
    const updatedSettings = prefillClassificationSettings(newSettings);
    useAppState.setState((state) => ({
        ...state,
        modelSettings: { ...state.modelSettings, ...updatedSettings } as ModelSettings,
    }));
}

export const resetTrainingReport = () => {
    useAppState.setState((prev: State) => ({ ...prev, report: initState.report }));
};

export const setTrainingStatus = (trainingState: TrainingState) => {
    useAppState.setState((prev: State) => ({ ...prev, trainingState }));
};

export const setPendingAction = (pendingAction: PendingAction) => {
    useAppState.setState((prev: State) => ({ ...prev, pendingAction }));
};

export const setTrainingReport = (report: TrainingReport) => {
    useAppState.setState((prev: State) => ({ ...prev, report }));
};
