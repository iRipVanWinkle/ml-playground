import type {
    ModelType,
    State,
    TaskType,
    TrainingReport,
    TrainingState,
    PendingAction,
    ModelSettings,
    LossFunction,
    ThetaInitializationConfig,
} from './types';
import {
    calculateMinMax,
    extractFeaturesAndLabels,
    generateCartesianProduct,
    labelEncoding,
} from './data/utils';
import { initState, useAppState } from './state';
import { modelSettingsDefaults } from './defaults';
import { readCsv } from '../shared/utils';
import { shuffleArray } from './data/shuffle';
import { useSystemSettings } from '../features/system-settings';

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
        let lossType: LossFunction = 'binaryCrossentropy';
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

export type ExtractFeaturesOptions = {
    file: File;
    shuffleData?: boolean;
    trainTestSplit?: number;
    taskType?: TaskType;
};

export async function extractFeatures({
    file,
    shuffleData,
    trainTestSplit,
    taskType,
}: ExtractFeaturesOptions) {
    const rawData = await readCsv(file);

    if (rawData.length === 0) {
        throw new Error('The CSV file is empty or not properly formatted.');
    }

    const headers = rawData.shift()!.map(String);

    let categories: string[] | undefined;
    if (taskType === 'classification') {
        categories = labelEncoding(rawData); // Convert string labels to numeric
    }

    if (shuffleData) {
        // Shuffle the data randomly
        const seed = useSystemSettings.getState().randomSeed;

        shuffleArray(rawData, seed);
    }

    const splitIndex = Math.floor(((trainTestSplit || 1) / 100) * rawData.length);

    const trainData = rawData.slice(0, splitIndex);
    const testData = rawData.slice(splitIndex);

    const { features: trainInputFeatures, labels: trainTargetLabels } =
        extractFeaturesAndLabels(trainData);
    const { features: testInputFeatures, labels: testTargetLabels } =
        extractFeaturesAndLabels(testData);

    const combinedFeatures = [...trainInputFeatures, ...testInputFeatures];
    const { xMin, xMax } = calculateMinMax(combinedFeatures);

    let predictionInputFeatures = undefined;
    if (trainInputFeatures[0].length < 3) {
        const predictionsNum = 150; // Number of points for predictions
        predictionInputFeatures = generateCartesianProduct(
            predictionsNum,
            xMin ?? [0],
            xMax ?? [0],
        );
    }

    useAppState.setState((state) => ({
        ...state,
        data: {
            trainInputFeatures,
            trainTargetLabels,
            testInputFeatures,
            testTargetLabels,
            predictionInputFeatures,
            xMin,
            xMax,
            headers,
            categories,
        },
    }));
}

export function resetData() {
    useAppState.setState((state) => ({
        ...state,
        data: initState.data,
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
