import { useMemo } from 'react';
import type { RocCurveData } from '../types';
import {
    diagonalLine,
    binaryLine,
    multiclassLine,
    thresholdMarkers,
    hasRocCurveData,
} from '../utils';

const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];

type UseRocCurvePlotDataParams = {
    rocCurveData: RocCurveData;
    categories?: string[];
};

export function useRocCurvePlotData({
    rocCurveData,
    categories,
}: UseRocCurvePlotDataParams): Plotly.Data[] {
    return useMemo(() => {
        if (!hasRocCurveData(rocCurveData)) return [diagonalLine()];

        const data: Plotly.Data[] = [];

        if (rocCurveData.type === 'binary') {
            // Binary classification: single curve
            const { fpr, tpr, thresholds, youdenOptimalIndex, closestToCornerIndex } = rocCurveData;
            const legendGroup = 'roc-binary';
            const color = colors[0];
            data.push(binaryLine({ x: fpr, y: tpr, thresholds, legendGroup, color }));
            data.push(
                ...thresholdMarkers({
                    fpr,
                    tpr,
                    thresholds,
                    youdenOptimalIndex,
                    closestToCornerIndex,
                    legendGroup,
                    color,
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
                const color = colors[index % colors.length];

                data.push(
                    multiclassLine({
                        x: fpr,
                        y: tpr,
                        thresholds,
                        label,
                        legendGroup,
                        color,
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
                        color,
                    }),
                );
            }
        }

        data.push(diagonalLine());

        return data;
    }, [rocCurveData, categories]);
}
