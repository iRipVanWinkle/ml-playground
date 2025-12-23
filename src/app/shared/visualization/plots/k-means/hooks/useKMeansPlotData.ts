import { useMemo } from 'react';
import type { MatrixLike } from '@/app/shared/helpers';

type ClustersType = {
    x: number[];
    y: number[];
    z: number[];
    indices: number[];
};

type UseKMeansPlotDataReturn = {
    trainClusterData: Map<number, ClustersType>;
    testClusterData?: Map<number, ClustersType>;
};

type UseKMeansPlotDataProps = {
    trainInputFeatures: number[][];
    testInputFeatures?: number[][];
    trainAssignments: MatrixLike;
    testAssignments?: MatrixLike;
};

export function useKMeansPlotData({
    trainInputFeatures,
    testInputFeatures,
    trainAssignments,
    testAssignments,
}: UseKMeansPlotDataProps): UseKMeansPlotDataReturn {
    return useMemo(() => {
        const hasTestData = testInputFeatures && testAssignments;

        return {
            trainClusterData: convertToClustersData(trainInputFeatures, trainAssignments),
            testClusterData: hasTestData
                ? convertToClustersData(testInputFeatures, testAssignments)
                : undefined,
        };
    }, [trainInputFeatures, testInputFeatures, trainAssignments, testAssignments]);
}

function convertToClustersData(inputFeatures: number[][], assignments: MatrixLike) {
    const clusterData = new Map<number, ClustersType>();

    for (let idx = 0; idx < inputFeatures.length; idx++) {
        const point = inputFeatures[idx];
        const clusterId = assignments.array[idx];

        if (!clusterData.has(clusterId)) {
            clusterData.set(clusterId, { x: [], y: [], z: [], indices: [] });
        }

        const cluster = clusterData.get(clusterId)!;

        cluster.indices.push(idx);
        cluster.x.push(point[0]);
        cluster.y.push(point[1]);
        if (point.length >= 3) {
            cluster.z.push(point[2]);
        }
    }

    return clusterData;
}
