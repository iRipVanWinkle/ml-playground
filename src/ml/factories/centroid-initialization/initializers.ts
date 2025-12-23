import { concat, type Tensor1D, type Tensor2D, tidy } from '@tensorflow/tfjs';
import { Randomizer } from '../../random/Randomizer';
import type { CentroidInitializer } from './types';

export function customCentroidInitFactory(centroids: Tensor2D): CentroidInitializer {
    return () => centroids.clone();
}

export function randomCentroidInit(X: Tensor2D, numClusters: number): Tensor2D {
    return tidy(() => {
        const numSamples = X.shape[0];

        const allIndices = Randomizer.randomUniqueNumber([numClusters], 0, numSamples, 'int32');

        return X.gather(allIndices);
    });
}

type ComputeDistancesType = (X: Tensor2D, centroids: Tensor2D) => Tensor2D;

export function kmeansPlusPlusCentroidInitFactory(
    computeDistances: ComputeDistancesType,
): CentroidInitializer {
    return (X: Tensor2D, numClusters: number): Tensor2D => {
        const [numSamples] = X.shape;

        return tidy(() => {
            const firstIdx = Randomizer.randomUniqueNumber([1], 0, numSamples, 'int32');

            let centroids = X.gather(firstIdx.squeeze());

            for (let i = 1; i < numClusters; i++) {
                // Compute distance to nearest centroid
                const distances = computeDistances(X, centroids);

                const minDist = distances.min(1);
                const minDistSquared = minDist.square();
                const probs = minDistSquared.div(minDistSquared.sum()) as Tensor1D;

                const nextIdx = sampleFromDistribution(probs);

                // Append centroid
                const nextCentroid = X.gather([nextIdx]);
                centroids = concat([centroids, nextCentroid], 0);
            }

            return centroids;
        });
    };
}

function sampleFromDistribution(probs: Tensor1D): number {
    // Convert cumulative distribution
    return tidy(() => {
        const cdf = probs.cumsum();

        const [r] = Randomizer.randomUniqueNumber([1], 0, 1).dataSync();
        const cdfArray = cdf.dataSync();

        for (let i = 0; i < cdfArray.length; i++) {
            if (r <= cdfArray[i]) {
                return i;
            }
        }

        return cdfArray.length - 1;
    });
}
