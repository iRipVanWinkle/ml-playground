import type { MatrixLike, TypedArray } from '@/ml/matrix';

export type KMeansMetricsData = {
    silhouetteSampleScores: TypedArray;
    silhouetteClusterScores: TypedArray;
    silhouetteScore: number;

    distanceToOtherCenters: MatrixLike;
    avgDistanceToOtherCenters: TypedArray;

    distanceToCenter: TypedArray;
    avgDistanceToCenter: TypedArray;
    maxDistanceToCenter: TypedArray;
};
