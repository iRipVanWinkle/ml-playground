import { readCsv } from '@/app/lib/readCsv';
import type {
    NormalizationFunction,
    ModelType,
    State,
    TaskType,
    TrainingReport,
    TrainingState,
    ClassificationType,
    PendingAction,
} from './types';
import {
    calculateMinMax,
    extractFeaturesAndLabels,
    generateCartesianProduct,
    labelEncoding,
} from './data/utils';
import { initState, useAppState } from './state';

export function setTaskType(taskType: TaskType) {
    useAppState.setState((state) => ({
        ...state,
        taskType,
    }));

    resetTrainingReport();
    resetData();
    setModelType(taskType === 'regression' ? 'linear' : 'logistic');
    if (taskType === 'classification') {
        setClassificationType('binary');
    }
}

export function setModelType(modelType: ModelType) {
    useAppState.setState((state) => ({
        ...state,
        modelSettings: {
            ...state.modelSettings,
            type: modelType,
            lossFunction: {
                type: state.taskType === 'regression' ? 'mse' : 'binaryCrossentropy',
            },
        },
    }));

    if (modelType === 'logistic') {
        setClassificationType('binary');
    }
}

export function setClassificationType(classificationType: ClassificationType) {
    useAppState.setState((state) => ({
        ...state,
        modelSettings: {
            ...state.modelSettings,
            classificationType,
            lossFunction: {
                type:
                    classificationType === 'softmax'
                        ? 'categoricalCrossentropy'
                        : 'binaryCrossentropy',
            },
        },
    }));
}

export function setNormalizationFunction(normalization: NormalizationFunction) {
    useAppState.setState((state) => ({
        ...state,
        dataSettings: {
            ...state.dataSettings,
            normalization,
        },
    }));
}

export function setTransformation(transformations: State['dataSettings']['transformations']) {
    useAppState.setState((state) => ({
        ...state,
        dataSettings: {
            ...state.dataSettings,
            transformations,
        },
    }));
}

export function updateModelSettings(newSettings: Partial<State['modelSettings']>) {
    useAppState.setState((state) => ({
        ...state,
        modelSettings: {
            ...state.modelSettings,
            ...newSettings,
        },
    }));
}

type ExtractFeaturesOptions = {
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
        rawData.sort(() => Math.random() - 0.5);
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
