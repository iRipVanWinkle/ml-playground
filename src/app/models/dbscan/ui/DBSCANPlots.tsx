import { useMemo } from 'react';
import type { Dataset } from '@/app/shared/types';
import type { DBSCANSettings, DBSCANTrainingReport } from '../types';
import { PlotlyScatter, PlotlyScatter3D } from '@/app/shared/visualization/plots/plotly';
import { useColor } from '@/app/shared/visualization/colors';

const NOISE_LABEL = -1;
const UNVISITED_LABEL = -2;

type DBSCANPlotsProps = {
    dataset: Dataset;
    report: DBSCANTrainingReport;
    modelSettings: DBSCANSettings;
};

type ClusterBucket2D = { x: number[]; y: number[] };
type ClusterBucket3D = { x: number[]; y: number[]; z: number[] };

function buildClusterBuckets2D(
    features: number[][],
    assignments: number[],
): Record<number, ClusterBucket2D> {
    const buckets: Record<number, ClusterBucket2D> = {};
    for (let i = 0; i < features.length; i++) {
        const label = assignments[i];
        if (!buckets[label]) buckets[label] = { x: [], y: [] };
        buckets[label].x.push(features[i][0]);
        buckets[label].y.push(features[i][1]);
    }
    return buckets;
}

function buildClusterBuckets3D(
    features: number[][],
    assignments: number[],
): Record<number, ClusterBucket3D> {
    const buckets: Record<number, ClusterBucket3D> = {};
    for (let i = 0; i < features.length; i++) {
        const label = assignments[i];
        if (!buckets[label]) buckets[label] = { x: [], y: [], z: [] };
        buckets[label].x.push(features[i][0]);
        buckets[label].y.push(features[i][1]);
        buckets[label].z.push(features[i][2]);
    }
    return buckets;
}

const LEGEND_LAYOUT = {
    x: 0.5,
    y: -0.2,
    xanchor: 'center' as const,
    yanchor: 'top' as const,
    orientation: 'h' as const,
};

const MARGIN = { l: 40, r: 40, t: 40, b: 40 };

export function DBSCANPlots({ dataset, report, modelSettings }: DBSCANPlotsProps) {
    const { trainInputFeatures, testInputFeatures, headers } = dataset;
    const { epsilon, distance } = modelSettings;
    const { trainAssignments, testAssignments, numClusters, activePointIndex } = report;
    const { getColor } = useColor();

    const is2DPlot = trainInputFeatures[0]?.length === 2;
    const is3DPlot = trainInputFeatures[0]?.length === 3;
    const hasAssignments = (trainAssignments?.shape[0] ?? 0) > 0;

    const [x1Label, x2Label, x3Label] = headers;

    const data2D = useMemo(() => {
        if (!is2DPlot) return [];

        if (!hasAssignments) {
            return [
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    mode: 'markers' as const,
                    name: 'Training Data',
                    marker: { color: 'grey', size: 8 },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    mode: 'markers' as const,
                    name: 'Test Data',
                    marker: { color: 'grey', size: 8, symbol: 'circle-open' as const },
                },
            ];
        }

        const trainBuckets = buildClusterBuckets2D(
            trainInputFeatures,
            Array.from(trainAssignments.array),
        );
        const testBuckets = testAssignments
            ? buildClusterBuckets2D(testInputFeatures, Array.from(testAssignments.array))
            : null;

        const traces = [];

        if (trainBuckets[UNVISITED_LABEL]) {
            traces.push({
                x: trainBuckets[UNVISITED_LABEL].x,
                y: trainBuckets[UNVISITED_LABEL].y,
                mode: 'markers' as const,
                name: 'Unvisited',
                marker: { color: 'grey', size: 8, symbol: 'circle' as const },
                legendgroup: 'unvisited',
            });
        }

        if (testBuckets?.[UNVISITED_LABEL]) {
            traces.push({
                x: testBuckets[UNVISITED_LABEL].x,
                y: testBuckets[UNVISITED_LABEL].y,
                mode: 'markers' as const,
                name: 'Unvisited (Test)',
                marker: { color: 'grey', size: 8, symbol: 'circle-open' as const },
                legendgroup: 'unvisited',
                showlegend: false,
            });
        }

        if (trainBuckets[NOISE_LABEL]) {
            traces.push({
                x: trainBuckets[NOISE_LABEL].x,
                y: trainBuckets[NOISE_LABEL].y,
                mode: 'markers' as const,
                name: 'Noise',
                marker: { color: 'grey', size: 5, symbol: 'x' as const },
                legendgroup: 'noise',
            });
        }

        if (testBuckets?.[NOISE_LABEL]) {
            traces.push({
                x: testBuckets[NOISE_LABEL].x,
                y: testBuckets[NOISE_LABEL].y,
                mode: 'markers' as const,
                name: 'Noise (Test)',
                marker: { color: 'grey', size: 5, symbol: 'x-open' as const },
                legendgroup: 'noise',
                showlegend: false,
            });
        }

        for (let c = 0; c < numClusters; c++) {
            if (trainBuckets[c]) {
                traces.push({
                    x: trainBuckets[c].x,
                    y: trainBuckets[c].y,
                    mode: 'markers' as const,
                    name: `Cluster ${c + 1}`,
                    marker: { color: getColor(c), size: 8, symbol: 'circle' as const },
                    legendgroup: `cluster-${c}`,
                });
            }
            if (testBuckets?.[c]) {
                traces.push({
                    x: testBuckets[c].x,
                    y: testBuckets[c].y,
                    mode: 'markers' as const,
                    name: `Cluster ${c + 1} (Test)`,
                    marker: { color: getColor(c), size: 8, symbol: 'circle-open' as const },
                    legendgroup: `cluster-${c}`,
                    showlegend: false,
                });
            }
        }

        // Active point highlight
        if (activePointIndex !== undefined && activePointIndex < trainInputFeatures.length) {
            traces.push({
                x: [trainInputFeatures[activePointIndex][0]],
                y: [trainInputFeatures[activePointIndex][1]],
                mode: 'markers' as const,
                name: 'Active Point',
                marker: {
                    color: 'rgba(0, 0, 0, 0)',
                    size: 16,
                    line: { color: '#ef4444', width: 3 },
                },
                showlegend: false,
            });
        }

        return traces;
    }, [
        is2DPlot,
        hasAssignments,
        trainInputFeatures,
        testInputFeatures,
        trainAssignments,
        testAssignments,
        numClusters,
        activePointIndex,
        getColor,
    ]);

    const data3D = useMemo(() => {
        if (!is3DPlot) return [];

        if (!hasAssignments) {
            return [
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    z: trainInputFeatures.map((p) => p[2]),
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: 'Training Data',
                    marker: { color: 'grey', size: 5 },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    z: testInputFeatures.map((p) => p[2]),
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: 'Test Data',
                    marker: { color: 'grey', size: 5, symbol: 'circle-open' as const },
                },
            ];
        }

        const trainBuckets = buildClusterBuckets3D(
            trainInputFeatures,
            Array.from(trainAssignments.array),
        );
        const testBuckets = testAssignments
            ? buildClusterBuckets3D(testInputFeatures, Array.from(testAssignments.array))
            : null;

        const traces = [];

        if (trainBuckets[UNVISITED_LABEL]) {
            traces.push({
                x: trainBuckets[UNVISITED_LABEL].x,
                y: trainBuckets[UNVISITED_LABEL].y,
                z: trainBuckets[UNVISITED_LABEL].z,
                mode: 'markers' as const,
                type: 'scatter3d' as const,
                name: 'Unvisited',
                marker: { color: 'grey', size: 5, symbol: 'circle' as const },
                legendgroup: 'unvisited',
            });
        }

        if (testBuckets?.[UNVISITED_LABEL]) {
            traces.push({
                x: testBuckets[UNVISITED_LABEL].x,
                y: testBuckets[UNVISITED_LABEL].y,
                z: testBuckets[UNVISITED_LABEL].z,
                mode: 'markers' as const,
                type: 'scatter3d' as const,
                name: 'Unvisited (Test)',
                marker: { color: 'grey', size: 5, symbol: 'circle-open' as const },
                legendgroup: 'unvisited',
                showlegend: false,
            });
        }

        if (trainBuckets[NOISE_LABEL]) {
            traces.push({
                x: trainBuckets[NOISE_LABEL].x,
                y: trainBuckets[NOISE_LABEL].y,
                z: trainBuckets[NOISE_LABEL].z,
                mode: 'markers' as const,
                type: 'scatter3d' as const,
                name: 'Noise',
                marker: { color: 'grey', size: 4, symbol: 'x' as const },
                legendgroup: 'noise',
            });
        }

        if (testBuckets?.[NOISE_LABEL]) {
            traces.push({
                x: testBuckets[NOISE_LABEL].x,
                y: testBuckets[NOISE_LABEL].y,
                z: testBuckets[NOISE_LABEL].z,
                mode: 'markers' as const,
                type: 'scatter3d' as const,
                name: 'Noise (Test)',
                marker: { color: 'grey', size: 4, symbol: 'x' as const },
                legendgroup: 'noise',
                showlegend: false,
            });
        }

        for (let c = 0; c <= numClusters; c++) {
            if (trainBuckets[c]) {
                traces.push({
                    x: trainBuckets[c].x,
                    y: trainBuckets[c].y,
                    z: trainBuckets[c].z,
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: `Cluster ${c + 1}`,
                    marker: { color: getColor(c), size: 5 },
                    legendgroup: `cluster-${c}`,
                });
            }
            if (testBuckets?.[c]) {
                traces.push({
                    x: testBuckets[c].x,
                    y: testBuckets[c].y,
                    z: testBuckets[c].z,
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: `Cluster ${c + 1} (Test)`,
                    marker: { color: getColor(c), size: 5, symbol: 'circle-open' as const },
                    legendgroup: `cluster-${c}`,
                    showlegend: false,
                });
            }
        }

        // Active point highlight
        if (activePointIndex !== undefined && activePointIndex < trainInputFeatures.length) {
            traces.push({
                x: [trainInputFeatures[activePointIndex][0]],
                y: [trainInputFeatures[activePointIndex][1]],
                z: [trainInputFeatures[activePointIndex][2]],
                mode: 'markers' as const,
                type: 'scatter3d' as const,
                name: 'Active Point',
                marker: {
                    color: '#ef4444',
                    size: 10,
                    line: { color: '#ef4444', width: 2 },
                },
                showlegend: false,
            });
        }

        return traces;
    }, [
        is3DPlot,
        hasAssignments,
        trainInputFeatures,
        testInputFeatures,
        trainAssignments,
        testAssignments,
        numClusters,
        activePointIndex,
        getColor,
    ]);

    const epsilonCircleShapes = useMemo(() => {
        if (
            !is2DPlot ||
            activePointIndex === undefined ||
            !epsilon ||
            !distance ||
            distance.type === 'cosine' ||
            activePointIndex >= trainInputFeatures.length
        ) {
            return [];
        }

        const cx = trainInputFeatures[activePointIndex][0];
        const cy = trainInputFeatures[activePointIndex][1];

        const sharedStyle = {
            xref: 'x' as const,
            yref: 'y' as const,
            line: { color: 'rgba(239, 68, 68, 0.4)', dash: 'dash' as const, width: 1.5 },
            fillcolor: 'rgba(239, 68, 68, 0.06)',
        };

        if (distance.type === 'euclidean') {
            return [
                {
                    type: 'circle' as const,
                    x0: cx - epsilon,
                    y0: cy - epsilon,
                    x1: cx + epsilon,
                    y1: cy + epsilon,
                    ...sharedStyle,
                },
            ];
        }

        // Manhattan: L1 ball is a diamond in data coordinates
        return [
            {
                type: 'path' as const,
                path: `M ${cx},${cy - epsilon} L ${cx + epsilon},${cy} L ${cx},${cy + epsilon} L ${cx - epsilon},${cy} Z`,
                ...sharedStyle,
            },
        ];
    }, [is2DPlot, activePointIndex, epsilon, distance, trainInputFeatures]);

    const axisRanges2D = useMemo(() => {
        if (!is2DPlot || trainInputFeatures.length === 0) return null;
        const allFeatures = [...trainInputFeatures, ...testInputFeatures];
        const xs = allFeatures.map((p) => p[0]);
        const ys = allFeatures.map((p) => p[1]);
        const xMin = Math.min(...xs);
        const xMax = Math.max(...xs);
        const yMin = Math.min(...ys);
        const yMax = Math.max(...ys);
        const eps = epsilon ?? 0;
        const xPad = (xMax - xMin) * 0.05 + eps;
        const yPad = (yMax - yMin) * 0.05 + eps;
        return {
            x: [xMin - xPad, xMax + xPad] as [number, number],
            y: [yMin - yPad, yMax + yPad] as [number, number],
        };
    }, [is2DPlot, trainInputFeatures, testInputFeatures, epsilon]);

    if (is2DPlot) {
        return (
            <PlotlyScatter
                data={data2D}
                layout={{
                    title: { text: 'DBSCAN Clustering' },
                    xaxis: {
                        title: { text: x1Label },
                        ...(axisRanges2D && { range: axisRanges2D.x, autorange: false }),
                    },
                    yaxis: {
                        title: { text: x2Label },
                        ...(axisRanges2D && { range: axisRanges2D.y, autorange: false }),
                    },
                    showlegend: true,
                    legend: LEGEND_LAYOUT,
                    margin: MARGIN,
                    shapes: epsilonCircleShapes,
                }}
            />
        );
    }

    if (is3DPlot) {
        return (
            <PlotlyScatter3D
                data={data3D}
                layout={{
                    title: { text: 'DBSCAN Clustering (3D)' },
                    scene: {
                        xaxis: { title: { text: x1Label } },
                        yaxis: { title: { text: x2Label } },
                        zaxis: { title: { text: x3Label } },
                    },
                    showlegend: true,
                    legend: LEGEND_LAYOUT,
                    margin: MARGIN,
                }}
            />
        );
    }

    return (
        <p className="text-sm text-muted-foreground p-4">Plotting requires 2 or 3 input features</p>
    );
}
