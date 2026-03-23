import { useMemo } from 'react';

type UseAxisRangesProps = {
    trainInputFeatures: number[][];
    testInputFeatures: number[][];
    epsilon: number;
};

export function useAxisRanges({
    trainInputFeatures,
    testInputFeatures,
    epsilon,
}: UseAxisRangesProps) {
    return useMemo(() => {
        if (trainInputFeatures.length === 0 || trainInputFeatures[0].length !== 2) return null;

        const allFeatures = [...trainInputFeatures, ...testInputFeatures];
        let xMin = Infinity,
            yMin = Infinity;
        let xMax = -Infinity,
            yMax = -Infinity;
        for (const feature of allFeatures) {
            xMin = Math.min(xMin, feature[0]);
            xMax = Math.max(xMax, feature[0]);
            yMin = Math.min(yMin, feature[1]);
            yMax = Math.max(yMax, feature[1]);
        }
        const eps = epsilon;
        const xPad = (xMax - xMin) * 0.05 + eps;
        const yPad = (yMax - yMin) * 0.05 + eps;

        return {
            x: [xMin - xPad, xMax + xPad] as [number, number],
            y: [yMin - yPad, yMax + yPad] as [number, number],
        };
    }, [epsilon, trainInputFeatures, testInputFeatures]);
}
