import type { ExtractFeaturesOptions } from '../types';
import { readCsv } from './csv-reader';
import { shuffleArray } from './shuffle';
import {
    calculateMinMax,
    extractFeaturesAndLabels,
    generateCartesianProduct,
    labelEncoding,
} from './transforms';

export async function extractFeatures({
    file,
    shuffleData,
    trainTestSplit,
    taskType,
    seed,
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

    return {
        trainInputFeatures,
        trainTargetLabels,
        testInputFeatures,
        testTargetLabels,
        predictionInputFeatures,
        xMin,
        xMax,
        headers,
        categories,
    };
}
