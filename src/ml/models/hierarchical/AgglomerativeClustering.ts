import { tensor2d, tidy, type Tensor2D } from '@tensorflow/tfjs';
import type {
    HierarchicalClusteringParams,
    TrainingControl,
    TrainingEventEmitter,
} from '../../types';
import { assert, range } from '../../utils';
import { euclideanDistance, type DistanceMetric } from '../../distance';
import { BaseHierarchicalClustering } from './BaseHierarchicalClustering';

export type LinkageMethod = 'single' | 'complete' | 'average' | 'ward';

type AgglomerativeClusteringModelOptions = {
    numClusters: number;
    linkage?: LinkageMethod;
    distanceMetric?: DistanceMetric;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

type LinkageParams = {
    dAX: number;
    dBX: number;
    sizeA: number;
    sizeB: number;
    sizeOther: number;
    mergeDistance: number;
};

/**
 * Agglomerative (bottom-up) hierarchical clustering.
 *
 * Starts with every point as its own singleton cluster, then iteratively
 * merges the two closest clusters according to the chosen `linkage`
 * criterion until `numClusters` clusters remain.
 *
 * Supported linkage methods:
 *  - `'single'`   – minimum pairwise distance (chaining-prone)
 *  - `'complete'` – maximum pairwise distance (compact clusters)
 *  - `'average'`  – UPGMA: weighted by cluster sizes
 *  - `'ward'`     – minimises total within-cluster variance (default)
 *
 * Cluster distances are updated at each merge step via the Lance-Williams
 * recurrence, so the n×n distance matrix is only computed once.
 *
 * Prediction for new points is performed by nearest-centroid assignment.
 */
export class AgglomerativeClustering extends BaseHierarchicalClustering {
    private linkage: LinkageMethod;
    private distanceMetric: DistanceMetric;

    constructor(options: AgglomerativeClusteringModelOptions) {
        assert(options.numClusters >= 2, 'Number of clusters must be at least 2.');
        super({
            numClusters: options.numClusters,
            eventEmitter: options.eventEmitter,
            trainingController: options.trainingController,
        });
        this.linkage = options.linkage ?? 'ward';
        this.distanceMetric = options.distanceMetric ?? euclideanDistance;
    }

    /**
     * Runs agglomerative clustering on `X`.
     *
     * @param X - Feature matrix of shape [n_samples, n_features].
     * @returns Trained model parameters (centroids + per-sample assignments).
     */
    async train(X: Tensor2D): Promise<HierarchicalClusteringParams> {
        const XArray = X.arraySync();
        const n = XArray.length;

        assert(
            this.numClusters <= n,
            `Number of clusters (${this.numClusters}) cannot exceed number of samples (${n})`,
        );

        this.iteration = 0;

        // Step 1: Initialize distance matrix and cluster assignments
        const distTensor = this.distanceMetric(X, X);
        const clusterDist = distTensor.arraySync();
        distTensor.dispose();

        // clusters[id] = point indices belonging to that cluster
        const clusters = new Map<number, number[]>();
        for (let i = 0; i < n; i++) {
            clusters.set(i, [i]);
        }

        // assignments[pointIndex] = current cluster id
        const assignments = new Int32Array(n);
        for (let i = 0; i < n; i++) {
            assignments[i] = i;
        }

        const activeClusters = new Set<number>(range(n));

        // Step 2: Iteratively merge the closest clusters
        while (activeClusters.size > this.numClusters) {
            if (this.trainingController?.isTrainingStopped) break;

            await this.trainingController?.handleControlFlow(true);

            // Find the pair of active clusters with the smallest linkage distance
            const [mergeA, mergeB, mergeDistance] = this.findClosestClusters(
                activeClusters,
                clusterDist,
            );

            const sizeA = clusters.get(mergeA)!.length;
            const sizeB = clusters.get(mergeB)!.length;

            // Merge B into A
            const merged = [...clusters.get(mergeA)!, ...clusters.get(mergeB)!];
            clusters.set(mergeA, merged);
            clusters.delete(mergeB);
            activeClusters.delete(mergeB);

            // Re-assign all points that belonged to B
            for (const pt of merged) {
                assignments[pt] = mergeA;
            }

            // Update linkage distances from the new merged cluster to every
            // remaining cluster via the Lance-Williams recurrence formula.
            for (const other of activeClusters) {
                if (other === mergeA) continue;

                const newDist = this.computeLinkageDistance({
                    dAX: clusterDist[mergeA][other],
                    dBX: clusterDist[mergeB][other],
                    sizeA,
                    sizeB,
                    sizeOther: clusters.get(other)!.length,
                    mergeDistance,
                });

                clusterDist[mergeA][other] = newDist;
                clusterDist[other][mergeA] = newDist;
            }

            await this.emitCallback(
                XArray,
                clusters,
                assignments,
                activeClusters.size,
                this.iteration++,
            );
        }

        // Step 3: Relabel cluster IDs to be contiguous (0 to k-1)
        this.relabelClusters(assignments, activeClusters);

        // Step 4: Compute final cluster centroids
        const k = activeClusters.size;
        const centroidsArr = this.computeCentroids(XArray, assignments, k);

        this.params = this.buildParams(centroidsArr, new Int32Array(assignments));

        return this.params;
    }

    protected assignPoints(X: Tensor2D, params: HierarchicalClusteringParams): number[] {
        const { centroids } = params;

        const distMatrix = tidy(() => {
            const centroidsTensor = tensor2d(centroids.array, centroids.shape);
            const distMatrix = this.distanceMetric(X, centroidsTensor);

            return distMatrix.argMin(1);
        });

        const result = distMatrix.arraySync() as number[];
        distMatrix.dispose();

        return result;
    }

    /**
     * Returns the closest pair of active clusters and their distance.
     *
     * @returns [mergeA, mergeB, distance]
     */
    private findClosestClusters(
        activeClusters: Set<number>,
        clusterDist: number[][],
    ): [number, number, number] {
        let minDist = Infinity;
        let mergeA = -1;
        let mergeB = -1;
        const activeArr = Array.from(activeClusters);

        for (let i = 0; i < activeArr.length; i++) {
            for (let j = i + 1; j < activeArr.length; j++) {
                const a = activeArr[i];
                const b = activeArr[j];
                const d = clusterDist[a][b];
                if (d < minDist) {
                    minDist = d;
                    mergeA = a;
                    mergeB = b;
                }
            }
        }

        return [mergeA, mergeB, minDist];
    }

    /**
     * Computes the updated linkage distance between the newly merged cluster
     * (A ∪ B) and another cluster X, using the Lance-Williams recurrence.
     */
    private computeLinkageDistance(p: LinkageParams): number {
        switch (this.linkage) {
            case 'single':
                return this.singleLinkage(p);
            case 'complete':
                return this.completeLinkage(p);
            case 'average':
                return this.averageLinkage(p);
            case 'ward':
                return this.wardLinkage(p);
        }
    }

    /** Minimum distance between any two points in the two clusters. */
    private singleLinkage({ dAX, dBX }: LinkageParams): number {
        return Math.min(dAX, dBX);
    }

    /** Maximum distance between any two points in the two clusters. */
    private completeLinkage({ dAX, dBX }: LinkageParams): number {
        return Math.max(dAX, dBX);
    }

    /**
     * UPGMA: weighted average by cluster sizes.
     * newDist = (|A| * d(A,X) + |B| * d(B,X)) / (|A| + |B|)
     */
    private averageLinkage({ dAX, dBX, sizeA, sizeB }: LinkageParams): number {
        return (sizeA * dAX + sizeB * dBX) / (sizeA + sizeB);
    }

    /**
     * Lance-Williams formula for Ward's minimum-variance criterion.
     * Distances are kept as Euclidean (not squared) to match the convention
     * used by the euclideanDistance utility.
     */
    private wardLinkage({
        dAX,
        dBX,
        sizeA,
        sizeB,
        sizeOther,
        mergeDistance,
    }: LinkageParams): number {
        return Math.sqrt(
            Math.max(
                0,
                ((sizeOther + sizeA) * dAX * dAX +
                    (sizeOther + sizeB) * dBX * dBX -
                    sizeOther * mergeDistance * mergeDistance) /
                    (sizeOther + sizeA + sizeB),
            ),
        );
    }
}
