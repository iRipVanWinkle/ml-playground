import { tidy, oneHot, type Tensor2D, type Tensor1D, where } from '@tensorflow/tfjs';
import { EPSILON } from '../../constants';
import { euclideanDistance, type DistanceMetric } from '../../distance';
import { isTensor2D } from '../../utils';

/**
 * Calculate distance from each point to its cluster center
 *
 * @param X               Tensor2D [N, D] — data
 * @param labels          Tensor2D [N, 1] — cluster assignments (int32)
 * @param centers         Tensor2D [K, D] — cluster centers
 * @param distanceMetric  DistanceMetric — function to compute distances
 * @returns Tensor1D [N] — distance from each point to its cluster center
 */
export function distanceToCenter(
    X: Tensor2D,
    labels: Tensor2D,
    centers: Tensor2D,
    distanceMetric: DistanceMetric = euclideanDistance,
): Tensor1D {
    return tidy(() => {
        const numClusters = centers.shape[0];
        const distances = distanceMetric(X, centers);

        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters);

        const distToCenter = distances.mul(oneHotMat).sum(1);

        return distToCenter as Tensor1D;
    });
}

/**
 * Calculate average distance to center per cluster
 *
 * @param distancesOrX     Either pre-calculated distances [N] OR data [N, D]
 * @param labels           Tensor2D [N, 1] — cluster assignments (int32)
 * @param centersOrNumClusters  Tensor2D [K, D] OR number — cluster centers if computing from raw data, or number of clusters if using pre-calculated distances
 * @param distanceMetric   DistanceMetric — only used if computing from raw data
 * @returns Tensor1D [K] — average distance to center for each cluster
 */
export function avgDistanceToCenter(
    distances: Tensor1D,
    labels: Tensor2D,
    numClusters: number,
): Tensor1D;
export function avgDistanceToCenter(
    X: Tensor2D,
    labels: Tensor2D,
    centers: Tensor2D,
    distanceMetric?: DistanceMetric,
): Tensor1D;
export function avgDistanceToCenter(
    distancesOrX: Tensor1D | Tensor2D,
    labels: Tensor2D,
    centersOrNumClusters: Tensor2D | number,
    distanceMetric?: DistanceMetric,
): Tensor1D {
    return tidy(() => {
        let pointDistances = distancesOrX as Tensor1D;

        if (isTensor2D(distancesOrX)) {
            pointDistances = distanceToCenter(
                distancesOrX,
                labels,
                centersOrNumClusters as Tensor2D,
                distanceMetric ?? euclideanDistance,
            );
        }

        const numClusters =
            typeof centersOrNumClusters === 'number'
                ? centersOrNumClusters
                : centersOrNumClusters.shape[0];

        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters);
        const clusterSizes = oneHotMat.sum(0);

        const clusterSum = pointDistances.expandDims(1).mul(oneHotMat).sum(0);
        const avgDist = clusterSum.div(clusterSizes.add(EPSILON));

        return avgDist as Tensor1D;
    });
}

/**
 * Calculate maximum distance to center per cluster
 *
 * @param distancesOrX     Either pre-calculated distances [N] OR data [N, D]
 * @param labels           Tensor2D [N, 1] — cluster assignments (int32)
 * @param centersOrNumClusters  Tensor2D [K, D] OR number — cluster centers if computing from raw data, or number of clusters if using pre-calculated distances
 * @param distanceMetric   DistanceMetric — only used if computing from raw data
 * @returns Tensor1D [K] — maximum distance to center for each cluster
 */
export function maxDistanceToCenter(
    distances: Tensor1D,
    labels: Tensor2D,
    numClusters: number,
): Tensor1D;
export function maxDistanceToCenter(
    X: Tensor2D,
    labels: Tensor2D,
    centers: Tensor2D,
    distanceMetric?: DistanceMetric,
): Tensor1D;
export function maxDistanceToCenter(
    distancesOrX: Tensor1D | Tensor2D,
    labels: Tensor2D,
    centersOrNumClusters: Tensor2D | number,
    distanceMetric?: DistanceMetric,
): Tensor1D {
    return tidy(() => {
        let pointDistances = distancesOrX as Tensor1D;

        if (isTensor2D(distancesOrX)) {
            pointDistances = distanceToCenter(
                distancesOrX,
                labels,
                centersOrNumClusters as Tensor2D,
                distanceMetric ?? euclideanDistance,
            );
        }

        const numClusters =
            typeof centersOrNumClusters === 'number'
                ? centersOrNumClusters
                : centersOrNumClusters.shape[0];

        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters);

        const maskedDistances = where(oneHotMat.toBool(), pointDistances.expandDims(1), -Infinity);
        const maxDist = maskedDistances.max(0);

        return maxDist as Tensor1D;
    });
}
