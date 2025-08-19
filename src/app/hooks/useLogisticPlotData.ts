import type { DataState } from '@/app/store';
import { useMemo } from 'react';

type GroupedLogisticPlotData = {
    trainingX: number[];
    trainingY: number[];
    testingX: number[];
    testingY: number[];
};

type GroupedPredictionPlotData = {
    predictionX: number[];
    predictionY: number[];
};

type LogisticPlotData = {
    groupedData: Record<number, GroupedLogisticPlotData>;
    groupedPredictions: GroupedPredictionPlotData | null;
};

export function useLogisticPlotData({
    trainTargetLabels,
    trainInputFeatures,
    testTargetLabels,
    testInputFeatures,
    predictionInputFeatures,
}: DataState): LogisticPlotData {
    return useMemo(() => {
        const groupedData: LogisticPlotData['groupedData'] = {};

        trainTargetLabels.forEach((label, index) => {
            const [x, y] = trainInputFeatures[index];
            const key = label[0];
            if (!groupedData[key]) {
                groupedData[key] = { trainingX: [], trainingY: [], testingX: [], testingY: [] };
            }
            groupedData[key].trainingX.push(x);
            groupedData[key].trainingY.push(y);
        });

        testTargetLabels.forEach((label, index) => {
            const [x, y] = testInputFeatures[index];
            const key = label[0];
            if (!groupedData[key]) {
                groupedData[key] = { trainingX: [], trainingY: [], testingX: [], testingY: [] };
            }
            groupedData[key].testingX.push(x);
            groupedData[key].testingY.push(y);
        });

        let groupedPredictions: LogisticPlotData['groupedPredictions'] = null;
        if (predictionInputFeatures) {
            const xSet = new Set<number>();
            const ySet = new Set<number>();

            for (const features of predictionInputFeatures) {
                xSet.add(features[0]);
                ySet.add(features[1]);
            }

            const predictionX = Array.from(xSet);
            const predictionY = Array.from(ySet);
            groupedPredictions = { predictionX, predictionY };
        }

        return { groupedData, groupedPredictions };
    }, [
        trainTargetLabels,
        testTargetLabels,
        predictionInputFeatures,
        trainInputFeatures,
        testInputFeatures,
    ]);
}
