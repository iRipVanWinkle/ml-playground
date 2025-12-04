import { useMemo } from 'react';
import type { RocCurveData } from '../types';
import {
    diagonalLine,
    binaryLine,
    multiclassLine,
    thresholdMarkers,
    hasRocCurveData,
} from '../utils';
import { useColor } from '../../../colors';

type UseRocCurvePlotDataParams = {
    rocCurveData: RocCurveData;
    categories?: string[];
};

export function useRocCurvePlotData({
    rocCurveData,
    categories,
}: UseRocCurvePlotDataParams): Plotly.Data[] {
    const { getColor } = useColor();

    return useMemo(() => {
        if (!hasRocCurveData(rocCurveData)) return [diagonalLine()];

        const data: Plotly.Data[] = [];

        if (rocCurveData.type === 'binary') {
            // Binary classification: single curve
            const { fpr, tpr, thresholds, youdenOptimalIndex, closestToCornerIndex } = rocCurveData;
            const legendGroup = 'roc-binary';
            data.push(binaryLine({ x: fpr, y: tpr, thresholds, legendGroup, color: getColor(0) }));
            data.push(
                ...thresholdMarkers({
                    fpr,
                    tpr,
                    thresholds,
                    youdenOptimalIndex,
                    closestToCornerIndex,
                    legendGroup,
                    color: getColor(0, 'darken'),
                }),
            );
        } else {
            // Multiclass classification: multiple curves (one per class)
            const { curves, classIndices } = rocCurveData;

            for (let index = 0; index < curves.length; index++) {
                const curve = curves[index];
                const { fpr, tpr, thresholds, youdenOptimalIndex, closestToCornerIndex } = curve;
                const classIndex = classIndices[index];
                const label = categories?.[classIndex] || `Class ${classIndex}`;
                const legendGroup = `roc-class-${classIndex}`;

                data.push(
                    multiclassLine({
                        x: fpr,
                        y: tpr,
                        thresholds,
                        label,
                        legendGroup,
                        color: getColor(index),
                    }),
                );

                data.push(
                    ...thresholdMarkers({
                        fpr,
                        tpr,
                        thresholds,
                        youdenOptimalIndex,
                        closestToCornerIndex,
                        legendGroup,
                        color: getColor(index, 'darken'),
                    }),
                );
            }
        }

        data.push(diagonalLine());

        return data;
    }, [rocCurveData, categories, getColor]);
}
