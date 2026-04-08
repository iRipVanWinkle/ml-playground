import type { TypedArray } from '@/app/shared/helpers';

type BinaryLineParams = {
    x: TypedArray;
    y: TypedArray;
    thresholds: TypedArray;
    legendGroup: string;
    color: string;
};

type MulticlassLineParams = {
    x: TypedArray;
    y: TypedArray;
    thresholds: TypedArray;
    label: string;
    legendGroup: string;
    color: string;
};

type OptimalThresholdMarkerParams = {
    x: number;
    y: number;
    optimalThreshold: number;
    label: string;
    legendGroup: string;
    color: string;
};

type ThresholdMarkersParams = {
    fpr: TypedArray;
    tpr: TypedArray;
    thresholds: TypedArray;
    youdenOptimalIndex: number | null;
    closestToCornerIndex: number | null;
    legendGroup: string;
    color: string;
};

/**
 * A line representing the diagonal line of the ROC curve.
 */
export const diagonalLine = () => ({
    x: [0, 1],
    y: [0, 1],
    mode: 'lines' as const,
    name: 'Random Classifier',
    line: {
        color: '#888888',
        dash: 'dash',
        width: 1,
    },
    showlegend: true,
    hoverinfo: 'skip' as const,
});

/**
 * A line representing the ROC curve for binary classification.
 */
export const binaryLine = ({ x, y, thresholds, legendGroup, color }: BinaryLineParams) => ({
    x,
    y,
    mode: 'lines' as const,
    name: `ROC`,
    line: {
        color,
        width: 2,
    },
    fill: 'tozeroy' as const,
    fillcolor: `${color}20`, // add opacity
    customdata: thresholds,
    legendgroup: legendGroup,
    hovertemplate:
        '<b>%{fullData.name}</b><br>' +
        'FPR: %{x:.3f}<br>' +
        'TPR: %{y:.3f}<br>' +
        'Threshold: %{customdata:.3f}<extra></extra>',
});

/**
 * A line representing the ROC curve for multiclass classification.
 */
export const multiclassLine = ({
    x,
    y,
    thresholds,
    label,
    legendGroup,
    color,
}: MulticlassLineParams) => ({
    x,
    y,
    mode: 'lines' as const,
    name: `${label}`,
    line: {
        color,
        width: 2,
    },
    customdata: thresholds,
    legendgroup: legendGroup,
    hovertemplate:
        `<b>%{fullData.name}</b><br>` +
        'FPR: %{x:.3f}<br>' +
        'TPR: %{y:.3f}<br>' +
        'Threshold: %{customdata:.3f}<extra></extra>',
});

/**
 * A line representing the threshold markers for the ROC curve.
 */
export function thresholdMarkers({
    fpr,
    tpr,
    thresholds,
    youdenOptimalIndex,
    closestToCornerIndex,
    legendGroup,
    color,
}: ThresholdMarkersParams): Plotly.Data[] {
    const result = [];
    const bothSameIndex = youdenOptimalIndex === closestToCornerIndex;

    if (youdenOptimalIndex != null) {
        const threshold = thresholds[youdenOptimalIndex];
        const label = getThresholdLabel('youden', threshold, bothSameIndex);

        result.push(
            optimalThresholdMarker({
                x: fpr[youdenOptimalIndex],
                y: tpr[youdenOptimalIndex],
                optimalThreshold: threshold,
                label,
                legendGroup,
                color,
            }),
        );
    }

    if (closestToCornerIndex != null && !bothSameIndex) {
        const threshold = thresholds[closestToCornerIndex];
        const label = getThresholdLabel('corner', threshold, false);

        result.push(
            optimalThresholdMarker({
                x: fpr[closestToCornerIndex],
                y: tpr[closestToCornerIndex],
                optimalThreshold: threshold,
                label,
                legendGroup,
                color,
            }),
        );
    }

    return result;
}

/**
 * A function to get the label for the threshold markers.
 */
function getThresholdLabel(
    type: 'youden' | 'corner',
    threshold: number,
    isBothOptimal: boolean,
): string {
    const formattedThreshold = threshold.toFixed(3);

    if (isBothOptimal) {
        return `Optimal Threshold (${formattedThreshold})`;
    }

    return type === 'youden'
        ? `Youden's J Threshold (${formattedThreshold})`
        : `Nearest Perfect Threshold (${formattedThreshold})`;
}

/**
 * A function to create the optimal threshold marker.
 */
const optimalThresholdMarker = ({
    x,
    y,
    optimalThreshold,
    label,
    legendGroup,
    color,
}: OptimalThresholdMarkerParams) => ({
    x: [x],
    y: [y],
    mode: 'markers' as const,
    name: label,
    marker: {
        color,
        size: 10,
        symbol: 'circle',
        line: {
            color: '#ffffff',
            width: 2,
        },
    },
    customdata: [optimalThreshold],
    legendgroup: legendGroup,
    showlegend: false,
    hovertemplate:
        `<b>%{fullData.name}</b><br>` +
        'FPR: %{x:.3f}<br>' +
        'TPR: %{y:.3f}<br>' +
        'Threshold: %{customdata:.3f}<extra></extra>',
});
