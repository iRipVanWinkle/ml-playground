import type { Tensor2D } from '@tensorflow/tfjs';
import type { KMeansMetricsData } from '../type';
import type { DistanceMetric } from '@/ml/distance';
import {
    avgDistanceToCenter,
    avgDistanceToOtherCenters,
    distanceToCenter,
    distanceToOtherCenters,
    maxDistanceToCenter,
    silhouetteCluster,
    silhouetteSample,
    silhouetteScore,
} from '@/ml/metrics';
import {
    getSafeMatrixFromTensor,
    getSafeTensorTypedArray,
    getSafeTensorValue,
} from '@/app/shared/workers';

export async function kMeansMetricData(
    X: Tensor2D,
    assignments: Tensor2D,
    centers: Tensor2D,
    distanceMetric?: DistanceMetric,
): Promise<KMeansMetricsData> {
    const numClusters = centers.shape[0];
    const silhouetteSampleScoresTensor = silhouetteSample(
        X,
        assignments,
        numClusters,
        distanceMetric,
    );
    const silhouetteClusterScoresTensor = silhouetteCluster(
        silhouetteSampleScoresTensor,
        assignments,
        numClusters,
    );
    const silhouetteScoreTensor = silhouetteScore(silhouetteClusterScoresTensor);

    const distanceToOtherCentersTensor = distanceToOtherCenters(
        X,
        assignments,
        centers,
        distanceMetric,
    );
    const avgDistanceToOtherCentersTensor = avgDistanceToOtherCenters(
        distanceToOtherCentersTensor,
        assignments,
        numClusters,
    );

    const distanceToCenterTensor = distanceToCenter(X, assignments, centers, distanceMetric);
    const avgDistanceToCenterTensor = avgDistanceToCenter(
        distanceToCenterTensor,
        assignments,
        numClusters,
    );
    const maxDistanceToCenterTensor = maxDistanceToCenter(
        distanceToCenterTensor,
        assignments,
        numClusters,
    );
    const [
        silhouetteSampleScores,
        silhouetteClusterScores,
        overallSilhouetteScore,
        otherCentersDistance,
        otherCentersDistanceAvg,

        centerDistance,
        centerDistanceAvg,
        centerDistanceMax,
    ] = await Promise.all([
        getSafeTensorTypedArray(silhouetteSampleScoresTensor),
        getSafeTensorTypedArray(silhouetteClusterScoresTensor),
        getSafeTensorValue(silhouetteScoreTensor),

        getSafeMatrixFromTensor(distanceToOtherCentersTensor),
        getSafeTensorTypedArray(avgDistanceToOtherCentersTensor),
        getSafeTensorTypedArray(distanceToCenterTensor),
        getSafeTensorTypedArray(avgDistanceToCenterTensor),
        getSafeTensorTypedArray(maxDistanceToCenterTensor),
    ]);

    return {
        silhouetteSampleScores,
        silhouetteClusterScores,
        silhouetteScore: overallSilhouetteScore,

        distanceToOtherCenters: otherCentersDistance,
        avgDistanceToOtherCenters: otherCentersDistanceAvg,

        distanceToCenter: centerDistance,
        avgDistanceToCenter: centerDistanceAvg,
        maxDistanceToCenter: centerDistanceMax,
    };
}
