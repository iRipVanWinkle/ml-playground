import {
    tidy,
    oneHot,
    maximum,
    eye,
    type Tensor2D,
    type Tensor1D,
    type Scalar,
    where,
} from '@tensorflow/tfjs';
import { EPSILON } from '../../constants';
import { euclideanDistance, type DistanceMetric } from '../../distance';
import { isTensor2D } from '../../utils';

/**
 * Silhouette score per sample
 *
 * @param X               Tensor2D [N, D] — data
 * @param labels          Tensor2D [N, 1] — cluster assignments (int32)
 * @param numClusters     number — number of clusters (K)
 * @param distanceMetric  DistanceMetric — function to compute pairwise distances
 * @returns Tensor1D [N] — silhouette score per sample
 */
export function silhouetteSample(
    X: Tensor2D,
    labels: Tensor2D,
    numClusters: number,
    distanceMetric: DistanceMetric = euclideanDistance,
): Tensor1D {
    return tidy(() => {
        const N = X.shape[0];
        const distances = distanceMetric(X, X);

        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters).toFloat();

        const sameCluster = oneHotMat.matMul(oneHotMat.transpose());

        // remove self-distance
        const maskIntra = sameCluster.sub(eye(N));
        const clusterCounts = maskIntra.sum(1);

        const a = distances.mul(maskIntra).sum(1).div(clusterCounts.add(EPSILON));

        const clusterDistSum = distances.matMul(oneHotMat);
        const clusterSizes = oneHotMat.sum(0);

        const meanClusterDist = clusterDistSum.div(clusterSizes.add(EPSILON));

        // exclude own cluster
        const masked = where(oneHotMat.toBool(), Infinity, meanClusterDist);
        const b = masked.min(1);

        const s = b.sub(a).div(maximum(a, b).add(EPSILON));

        return s as Tensor1D;
    });
}

/**
 * Silhouette score per cluster
 *
 * @param sampleScoresOrX  Either pre-calculated sample scores [N] OR data [N, D]
 * @param labels           Tensor2D [N, 1] — cluster assignments (int32)
 * @param numClusters      number — number of clusters (K)
 * @param distanceMetric   DistanceMetric — only used if computing from raw data
 * @returns Tensor1D [K] — silhouette score per cluster
 */
export function silhouetteCluster(
    sampleScores: Tensor1D,
    labels: Tensor2D,
    numClusters: number,
): Tensor1D;
export function silhouetteCluster(
    X: Tensor2D,
    labels: Tensor2D,
    numClusters: number,
    distanceMetric?: DistanceMetric,
): Tensor1D;
export function silhouetteCluster(
    sampleScoresOrX: Tensor1D | Tensor2D,
    labels: Tensor2D,
    numClusters: number,
    distanceMetric?: DistanceMetric,
): Tensor1D {
    return tidy(() => {
        let sampleScores = sampleScoresOrX;

        if (isTensor2D(sampleScoresOrX)) {
            sampleScores = silhouetteSample(
                sampleScoresOrX,
                labels,
                numClusters,
                distanceMetric ?? euclideanDistance,
            );
        }
        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters);
        const clusterSizes = oneHotMat.sum(0);

        const clusterSum = sampleScores.expandDims(1).mul(oneHotMat).sum(0);
        const clusterScores = clusterSum.div(clusterSizes.add(EPSILON));

        return clusterScores as Tensor1D;
    });
}

/**
 * Overall silhouette score (mean across all samples)
 *
 * @param sampleOrClusterScoresOrX  Either sample scores [N], cluster scores [K], OR data [N, D]
 * @param labels                    Tensor2D [N, 1] — required if computing from raw data
 * @param numClusters               number — required if computing from raw data
 * @param distanceMetric            DistanceMetric — only used if computing from raw data
 * @returns Scalar — overall silhouette score
 */
export function silhouetteScore(sampleOrClusterScores: Tensor1D): Scalar;
export function silhouetteScore(
    X: Tensor2D,
    labels: Tensor2D,
    numClusters: number,
    distanceMetric?: DistanceMetric,
): Scalar;
export function silhouetteScore(
    sampleOrClusterScoresOrX: Tensor1D | Tensor2D,
    labels?: Tensor2D,
    numClusters?: number,
    distanceMetric?: DistanceMetric,
): Scalar {
    return tidy(() => {
        let scores = sampleOrClusterScoresOrX;

        if (isTensor2D(sampleOrClusterScoresOrX)) {
            if (!labels || !numClusters) {
                throw new Error(
                    'Labels and numClusters are required when computing silhouette score from raw data.',
                );
            }

            scores = silhouetteSample(
                sampleOrClusterScoresOrX,
                labels,
                numClusters,
                distanceMetric ?? euclideanDistance,
            );
        }

        return scores.mean();
    });
}
