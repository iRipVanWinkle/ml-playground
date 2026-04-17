import type { MatrixLike, TypedArray } from '@/app/shared/helpers';

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
