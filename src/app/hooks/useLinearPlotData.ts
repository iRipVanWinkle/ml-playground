import type { DataState } from '@/app/store';
import { useMemo } from 'react';

export function useLinearPlotData({
    trainInputFeatures,
    trainTargetLabels,
    testInputFeatures,
    testTargetLabels,
    predictionInputFeatures,
}: DataState) {
    return useMemo(() => {
        const is2DPlot = trainInputFeatures[0]?.length === 1;
        const is3DPlot = trainInputFeatures[0]?.length === 2;

        if (is2DPlot) {
            return {
                trainX: trainInputFeatures?.flat(),
                trainY: trainTargetLabels?.flat(),
                testX: testInputFeatures?.flat(),
                testY: testTargetLabels?.flat(),
                predictionX: predictionInputFeatures?.flat(),
            };
        }

        if (is3DPlot) {
            return {
                trainX: trainInputFeatures.map((f) => f[0]),
                trainY: trainInputFeatures.map((f) => f[1]),
                trainZ: trainTargetLabels?.flat(),
                testX: testInputFeatures.map((f) => f[0]),
                testY: testInputFeatures.map((f) => f[1]),
                testZ: testTargetLabels?.flat(),
                predictionX: (predictionInputFeatures ?? []).map((p) => p[0]),
                predictionY: (predictionInputFeatures ?? []).map((p) => p[1]),
            };
        }

        return {};
    }, [
        trainInputFeatures,
        trainTargetLabels,
        testInputFeatures,
        testTargetLabels,
        predictionInputFeatures,
    ]);
}
