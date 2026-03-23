import type { Dataset } from '@/app/shared/types';
import type { KMeansTrainingReport } from '@/app/models/k-means/types';
import { PlotlyScatter, PlotlyScatter3D } from '../plotly';
import { useColor } from '../../colors';
import { useMemo } from 'react';
import { useKMeansPlotData } from './hooks/useKMeansPlotData';

type KMeansPlotsProps = {
    dataset: Dataset;
    report: KMeansTrainingReport;
};

export function KMeansPlots({ dataset, report }: KMeansPlotsProps) {
    const { centroids, trainAssignments, testAssignments } = report;
    const { trainInputFeatures, testInputFeatures, headers } = dataset;

    const { getColor } = useColor();

    const isKMeans = report.type === 'k-means';
    const is2DPlot = trainInputFeatures[0]?.length === 2;
    const is3DPlot = trainInputFeatures[0]?.length === 3;

    const { trainClusterData, testClusterData } = useKMeansPlotData({
        trainInputFeatures,
        testInputFeatures,
        trainAssignments,
        testAssignments,
    });

    const centroidData = useMemo(() => {
        if (!centroids?.array || centroids.array.length === 0 || !centroids.shape) {
            return { x: [], y: [], z: [] };
        }

        const [numCentroids, numFeatures] = centroids.shape;
        const x: number[] = [];
        const y: number[] = [];
        const z: number[] = [];

        for (let i = 0; i < numCentroids; i++) {
            x.push(centroids.array[i * numFeatures]);
            y.push(centroids.array[i * numFeatures + 1]);
            if (is3DPlot) {
                z.push(centroids.array[i * numFeatures + 2]);
            }
        }

        return { x, y, z };
    }, [centroids, is3DPlot]);

    const [x1Label, x2Label, x3Label] = headers;

    const hasValidClusters = isKMeans && trainClusterData && centroidData.x.length > 0;

    if (is2DPlot) {
        const plotData = [];

        if (hasValidClusters) {
            plotData.push(
                ...Array.from(trainClusterData.entries()).map(([clusterId, points]) => {
                    return {
                        x: points.x,
                        y: points.y,
                        mode: 'markers' as const,
                        name: `Cluster ${clusterId}`,
                        marker: {
                            color: getColor(clusterId),
                            size: 8,
                        },
                        legendgroup: `cluster-${clusterId}`,
                    };
                }),
            );

            if (testClusterData) {
                plotData.push(
                    ...Array.from(testClusterData.entries()).map(([clusterId, points]) => {
                        return {
                            x: points.x,
                            y: points.y,
                            mode: 'markers' as const,
                            name: `Cluster ${clusterId} (Test)`,
                            marker: {
                                color: getColor(clusterId),
                                size: 8,
                                symbol: 'circle-open',
                            },
                            legendgroup: `cluster-${clusterId}`,
                            showlegend: false,
                        };
                    }),
                );
            }

            plotData.push({
                x: centroidData.x,
                y: centroidData.y,
                mode: 'markers' as const,
                name: 'Centroids',
                marker: {
                    color: 'black',
                    size: 16,
                    symbol: 'x',
                    line: {
                        color: 'white',
                        width: 2,
                    },
                },
            });
        } else {
            plotData.push(
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    mode: 'markers' as const,
                    name: 'Training Data',
                    marker: {
                        color: 'grey',
                        size: 8,
                    },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    mode: 'markers' as const,
                    name: 'Test Data',
                    marker: {
                        color: 'grey',
                        size: 8,
                        line: {
                            color: 'grey',
                            width: 2,
                        },
                        symbol: 'circle-open',
                    },
                },
            );
        }

        return (
            <PlotlyScatter
                data={plotData}
                layout={{
                    title: { text: 'K-Means Clustering' },
                    xaxis: { title: { text: x1Label } },
                    yaxis: { title: { text: x2Label } },
                    showlegend: true,
                    legend: {
                        x: 0.5,
                        y: -0.2,
                        xanchor: 'center',
                        yanchor: 'top',
                        orientation: 'h',
                    },
                    margin: { l: 40, r: 40, t: 40, b: 40 },
                }}
            />
        );
    }

    if (is3DPlot) {
        const plotData = [];

        if (hasValidClusters) {
            plotData.push(
                ...Array.from(trainClusterData.entries()).map(([clusterId, points]) => {
                    return {
                        x: points.x,
                        y: points.y,
                        z: points.z,
                        mode: 'markers' as const,
                        type: 'scatter3d' as const,
                        name: `Cluster ${clusterId}`,
                        marker: {
                            color: getColor(clusterId),
                            size: 5,
                        },
                        legendgroup: `cluster-${clusterId}`,
                    };
                }),
            );

            if (testClusterData) {
                plotData.push(
                    ...Array.from(testClusterData.entries()).map(([clusterId, points]) => {
                        return {
                            x: points.x,
                            y: points.y,
                            z: points.z,
                            mode: 'markers' as const,
                            type: 'scatter3d' as const,
                            name: `Cluster ${clusterId} (Test)`,
                            marker: {
                                color: getColor(clusterId),
                                size: 5,
                                symbol: 'circle-open',
                            },
                            legendgroup: `cluster-${clusterId}`,
                            showlegend: false,
                        };
                    }),
                );
            }

            plotData.push({
                x: centroidData.x,
                y: centroidData.y,
                z: centroidData.z,
                mode: 'markers' as const,
                type: 'scatter3d' as const,
                name: 'Centroids',
                marker: {
                    color: 'black',
                    size: 12,
                    symbol: 'diamond',
                    line: {
                        color: 'white',
                        width: 2,
                    },
                },
            });
        } else {
            plotData.push(
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    z: trainInputFeatures.map((p) => p[2]),
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: 'Training Data',
                    marker: {
                        color: 'grey',
                        size: 5,
                    },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    z: testInputFeatures.map((p) => p[2]),
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: 'Test Data',
                    marker: {
                        color: 'grey',
                        size: 5,
                        line: {
                            color: 'grey',
                            width: 2,
                        },
                        symbol: 'circle-open',
                    },
                },
            );
        }

        return (
            <PlotlyScatter3D
                data={plotData}
                layout={{
                    title: { text: 'K-Means Clustering (3D)' },
                    scene: {
                        xaxis: { title: { text: x1Label } },
                        yaxis: { title: { text: x2Label } },
                        zaxis: { title: { text: x3Label } },
                    },
                    showlegend: true,
                    legend: {
                        x: 0.5,
                        y: -0.2,
                        xanchor: 'center',
                        yanchor: 'top',
                        orientation: 'h',
                    },
                    margin: { l: 40, r: 40, t: 40, b: 40 },
                }}
            />
        );
    }

    return (
        <p className="text-sm text-muted-foreground p-4">Plotting requires 2 or 3 input features</p>
    );
}
