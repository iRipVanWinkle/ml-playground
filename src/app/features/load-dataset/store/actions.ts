import type { ExtractFeaturesOptions } from './types';
import { readCsv } from '../libs/csv-reader';
import { shuffleArray } from '../libs/shuffle';
import { useSystemSettings } from '../../system-settings';
import {
    calculateMinMax,
    extractFeaturesAndLabels,
    generateCartesianProduct,
    labelEncoding,
} from '../libs/transforms';
import { useDataset, initState } from './store';

export function resetData() {
    useDataset.setState(initState);
}

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

    let predictionInputFeatures: number[][] | undefined;
    if (trainInputFeatures[0].length < 3) {
        const predictionsNum = 150; // Number of points for predictions
        predictionInputFeatures = generateCartesianProduct(
            predictionsNum,
            xMin ?? [0],
            xMax ?? [0],
        );
    }

    useDataset.setState((prev) => ({
        ...prev,
        trainInputFeatures,
        trainTargetLabels,
        testInputFeatures,
        testTargetLabels,
        predictionInputFeatures,
        xMin,
        xMax,
        headers,
        categories,
    }));
}
