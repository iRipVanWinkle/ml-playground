import type { MatrixLike } from '@/app/shared/helpers';
import { useMemo } from 'react';

const NOISE_LABEL = -1;
const UNVISITED_LABEL = -2;

type UseDBSCANPlotDataProps = {
    trainInputFeatures: number[][];
    testInputFeatures: number[][];
    trainPredictions?: MatrixLike;
    testPredictions?: MatrixLike;
    numClusters: number;
    activePointIndex?: number;
    hasAssignments: boolean;
    getColor: (clusterId: number) => string;
};

export function useDBSCANPlotData({
    trainInputFeatures,
    testInputFeatures,
    trainPredictions,
    testPredictions,
    numClusters,
    activePointIndex,
    hasAssignments,
    getColor,
}: UseDBSCANPlotDataProps) {
    return useMemo(() => {
        const is2DPlot = trainInputFeatures[0]?.length === 2;

        if (!is2DPlot) return [];

        if (!hasAssignments) {
            return [
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    mode: 'markers',
                    name: 'Training Data',
                    marker: { color: 'grey', size: 8 },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    mode: 'markers',
                    name: 'Test Data',
                    marker: { color: 'grey', size: 8, symbol: 'circle-open' },
                },
            ];
        }

        const trainBuckets = buildClusterBuckets(trainInputFeatures, trainPredictions!.array);
        const testBuckets = testPredictions
            ? buildClusterBuckets(testInputFeatures, testPredictions.array)
            : {};

        const traces = [];

        if (trainBuckets[UNVISITED_LABEL]) {
            traces.push({
                x: trainBuckets[UNVISITED_LABEL].x,
                y: trainBuckets[UNVISITED_LABEL].y,
                mode: 'markers',
                name: 'Unvisited',
                marker: { color: 'grey', size: 8, symbol: 'circle' },
                legendgroup: 'unvisited',
            });
        }

        if (testBuckets[UNVISITED_LABEL]) {
            traces.push({
                x: testBuckets[UNVISITED_LABEL].x,
                y: testBuckets[UNVISITED_LABEL].y,
                mode: 'markers',
                name: 'Unvisited (Test)',
                marker: { color: 'grey', size: 8, symbol: 'circle-open' },
                legendgroup: 'unvisited',
                showlegend: false,
            });
        }

        if (trainBuckets[NOISE_LABEL]) {
            traces.push({
                x: trainBuckets[NOISE_LABEL].x,
                y: trainBuckets[NOISE_LABEL].y,
                mode: 'markers',
                name: 'Noise',
                marker: { color: 'grey', size: 5, symbol: 'x' },
                legendgroup: 'noise',
            });
        }

        if (testBuckets[NOISE_LABEL]) {
            traces.push({
                x: testBuckets[NOISE_LABEL].x,
                y: testBuckets[NOISE_LABEL].y,
                mode: 'markers',
                name: 'Noise (Test)',
                marker: { color: 'grey', size: 5, symbol: 'x-open' },
                legendgroup: 'noise',
                showlegend: false,
            });
        }

        for (let clusterId = 0; clusterId < numClusters; clusterId++) {
            if (trainBuckets[clusterId]) {
                traces.push({
                    x: trainBuckets[clusterId].x,
                    y: trainBuckets[clusterId].y,
                    mode: 'markers',
                    name: `Cluster ${clusterId + 1}`,
                    marker: { color: getColor(clusterId), size: 8, symbol: 'circle' },
                    legendgroup: `cluster-${clusterId}`,
                });
            }
            if (testBuckets[clusterId]) {
                traces.push({
                    x: testBuckets[clusterId].x,
                    y: testBuckets[clusterId].y,
                    mode: 'markers',
                    name: `Cluster ${clusterId + 1}`,
                    marker: { color: getColor(clusterId), size: 8, symbol: 'circle-open' },
                    legendgroup: `cluster-${clusterId}`,
                    showlegend: false,
                });
            }
        }

        // Active point highlight
        if (activePointIndex !== undefined && activePointIndex < trainInputFeatures.length) {
            traces.push({
                x: [trainInputFeatures[activePointIndex][0]],
                y: [trainInputFeatures[activePointIndex][1]],
                mode: 'markers',
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
        trainInputFeatures,
        testInputFeatures,
        trainPredictions,
        testPredictions,
        numClusters,
        activePointIndex,
        hasAssignments,
        getColor,
    ]);
}

type ClusterBucket2D = { x: number[]; y: number[] };

function buildClusterBuckets(
    features: number[][],
    assignments: ArrayLike<number>,
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
