import { tidy, oneHot, type Tensor2D, type Tensor1D } from '@tensorflow/tfjs';
import { EPSILON } from '../../constants';
import { euclideanDistance, type DistanceMetric } from '../../distance';

/**
 * Calculate distances from each point to other cluster centers (not its own)
 *
 * @param X               Tensor2D [N, D] — data
 * @param labels          Tensor2D [N, 1] — cluster assignments (int32)
 * @param centers         Tensor2D [K, D] — cluster centers
 * @param distanceMetric  DistanceMetric — function to compute distances
 * @returns Tensor2D [N, K] — distances from each point to all centers (assigned center excluded)
 */
export function distanceToOtherCenters(
    X: Tensor2D,
    labels: Tensor2D,
    centers: Tensor2D,
    distanceMetric: DistanceMetric = euclideanDistance,
): Tensor2D {
    return tidy(() => {
        const numClusters = centers.shape[0];
        const distances = distanceMetric(X, centers);

        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters);
        const inverseMask = oneHotMat.mul(-1).add(1);

        const filtered = distances.mul(inverseMask);

        return filtered as Tensor2D;
    });
}

/**
 * Calculate average distance to other cluster centers per cluster
 *
 * @param distancesOrX     Either pre-calculated distances [N, K] OR data [N, D]
 * @param labels           Tensor2D [N, 1] — cluster assignments (int32)
 * @param centersOrNumClusters  Tensor2D [K, D] OR number — cluster centers if computing from raw data, or number of clusters if using pre-calculated distances
 * @param distanceMetric   DistanceMetric — only used if computing from raw data
 * @returns Tensor1D [K] — average distance to other centers for each cluster
 */
export function avgDistanceToOtherCenters(
    distances: Tensor2D,
    labels: Tensor2D,
    numClusters: number,
): Tensor1D;
export function avgDistanceToOtherCenters(
    X: Tensor2D,
    labels: Tensor2D,
    centers: Tensor2D,
    distanceMetric?: DistanceMetric,
): Tensor1D;
export function avgDistanceToOtherCenters(
    distancesOrX: Tensor2D,
    labels: Tensor2D,
    centersOrNumClusters: Tensor2D | number,
    distanceMetric?: DistanceMetric,
): Tensor1D {
    return tidy(() => {
        let distancesToOthers = distancesOrX;

        if (typeof centersOrNumClusters !== 'number') {
            distancesToOthers = distanceToOtherCenters(
                distancesOrX,
                labels,
                centersOrNumClusters,
                distanceMetric ?? euclideanDistance,
            );
        }

        const numClusters =
            typeof centersOrNumClusters === 'number'
                ? centersOrNumClusters
                : centersOrNumClusters.shape[0];

        const avgPointDistances = distancesToOthers.sum(1).div(numClusters - 1);

        const labelsFlat = labels.squeeze().toInt();
        const oneHotMat = oneHot(labelsFlat, numClusters);
        const clusterSizes = oneHotMat.sum(0);

        const clusterSum = avgPointDistances.expandDims(1).mul(oneHotMat).sum(0);
        const avgDist = clusterSum.div(clusterSizes.add(EPSILON));

        return avgDist as Tensor1D;
    });
}
