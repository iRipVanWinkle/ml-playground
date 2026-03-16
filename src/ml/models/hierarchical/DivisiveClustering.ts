import { type Rank, tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type {
    HierarchicalClusteringParams,
    TrainingControl,
    TrainingEventEmitter,
} from '../../types';
import { assert, gather, range } from '../../utils';
import {
    EuclideanClusteringMath,
    type DistanceFunction,
    type CentroidFunction,
} from '../../distance';
import { BaseHierarchicalClustering } from './BaseHierarchicalClustering';
import { Randomizer } from '../../random/Randomizer';

type DivisiveClusteringModelOptions = {
    numClusters: number;
    bisectIterations?: number;
    bisectRestarts?: number;
    distanceFunction?: DistanceFunction;
    centroidFunction?: CentroidFunction;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

/**
 * Divisive (top-down) hierarchical clustering.
 *
 * Starts with all points in a single cluster then repeatedly splits the
 * cluster with the highest SSE (sum of squared errors from its centroid)
 * using a bisecting k-means strategy until `numClusters` clusters remain.
 *
 * This is the standard DIANA-style approach adapted to produce exactly
 * `numClusters` groups, with prediction performed by nearest-centroid
 * assignment (using the provided `distanceMetric`).
 */
export class DivisiveClustering extends BaseHierarchicalClustering {
    private bisectIterations: number;
    private bisectRestarts: number;
    private distanceFunction: DistanceFunction;
    private centroidFunction: CentroidFunction;

    constructor(options: DivisiveClusteringModelOptions) {
        assert(options.numClusters >= 2, 'Number of clusters must be at least 2.');
        super({
            numClusters: options.numClusters,
            eventEmitter: options.eventEmitter,
            trainingController: options.trainingController,
        });

        this.bisectIterations = options.bisectIterations ?? 20;
        this.bisectRestarts = options.bisectRestarts ?? 3;

        const defaultMath = new EuclideanClusteringMath();
        this.distanceFunction = options.distanceFunction ?? defaultMath.distance;
        this.centroidFunction = options.centroidFunction ?? defaultMath.centroid;
    }

    /**
     * Runs divisive clustering on `X`.
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

        // Step 1: Initialize all points into a single starting cluster
        const assignments = new Int32Array(n);
        const clusters = new Map<number, number[]>([[0, range(n)]]);
        let nextId = 1;

        await this.emitCallback(XArray, clusters, assignments, clusters.size, clusters.size - 1);

        // Step 2: Iteratively split the cluster with the highest SSE
        while (clusters.size < this.numClusters) {
            await this.trainingController?.handleControlFlow(true);

            if (this.trainingController?.isTrainingStopped) break;

            // Pick the cluster with the highest SSE
            let worstId = -1;
            let worstSSE = -Infinity;

            for (const [id, indices] of clusters) {
                if (indices.length < 2) continue; // cannot split singletons

                const pts = gather(XArray, indices);
                const s = this.sse(pts);
                if (s > worstSSE) {
                    worstSSE = s;
                    worstId = id;
                }
            }

            if (worstId === -1) break; // all remaining clusters are singletons

            const clusterIndices = clusters.get(worstId)!;
            const clusterPoints = gather(XArray, clusterIndices);

            // Bisect the chosen cluster
            const [localA, localB] = this.bisect(clusterPoints);

            const globalA = gather(clusterIndices, localA);
            const globalB = gather(clusterIndices, localB);

            const newId = nextId++;

            // Update cluster map: keep worstId for groupA, create newId for groupB
            clusters.set(worstId, globalA);
            clusters.set(newId, globalB);

            // Update assignments for groupB points
            for (const pt of globalB) {
                assignments[pt] = newId;
            }

            await this.emitCallback(
                XArray,
                clusters,
                assignments,
                clusters.size,
                clusters.size - 1,
            );
        }

        // Step 3: Relabel cluster IDs to be contiguous (0 to k-1)
        this.relabelClusters(assignments, new Set(clusters.keys()));

        // Step 4: Compute final cluster centroids
        const k = clusters.size;
        const centroidsArr = this.computeCentroids(XArray, assignments, k);

        this.params = this.buildParams(centroidsArr, new Int32Array(assignments));

        return this.params;
    }

    protected assignPoints(X: Tensor2D, params: HierarchicalClusteringParams): number[] {
        const { centroids } = params;
        const XArray = X.arraySync() as number[][];

        const centroidsTensor = tensor2d(centroids.array, centroids.shape);
        const centroidsArray = centroidsTensor.arraySync() as number[][];
        centroidsTensor.dispose();

        return XArray.map((point) => {
            let bestCluster = 0;
            let bestDist = Infinity;
            for (let c = 0; c < centroidsArray.length; c++) {
                const d = this.distanceFunction(point, centroidsArray[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestCluster = c;
                }
            }
            return bestCluster;
        });
    }

    private sse(pts: number[][]): number {
        if (pts.length === 0) return 0;
        const c = this.centroidFunction(pts);
        let s = 0;

        for (const p of pts) {
            const dist = this.distanceFunction(p, c);
            s += dist * dist;
        }
        return s;
    }

    /**
     * Bisect `points` using 2-means with `maxIter` iterations and `restarts`
     * random initialisations.  Returns two disjoint arrays of *local* indices
     * (indices into `points`).
     */
    private bisect(points: number[][]): [number[], number[]] {
        const n = points.length;
        const distance = this.distanceFunction;
        const centroid = this.centroidFunction;

        // Trivial cases – no k-means required
        if (n <= 1) return [[0], []];
        if (n === 2) return [[0], [1]];

        let bestA: number[] = [];
        let bestB: number[] = [];
        let bestSSE = Infinity;

        for (let r = 0; r < this.bisectRestarts; r++) {
            const seedTensor = Randomizer.randomUniqueNumber<Rank.R1>([2], 0, n, 'int32');
            const seeds = seedTensor.arraySync();
            seedTensor.dispose();

            let cA = [...points[seeds[0]]];
            let cB = [...points[seeds[1]]];

            let gA: number[] = [];
            let gB: number[] = [];

            for (let iter = 0; iter < this.bisectIterations; iter++) {
                gA = [];
                gB = [];

                for (let i = 0; i < n; i++) {
                    if (distance(points[i], cA) <= distance(points[i], cB)) {
                        gA.push(i);
                    } else {
                        gB.push(i);
                    }
                }

                // Degenerate split – force a balanced initial split and stop
                if (gA.length === 0 || gB.length === 0) {
                    gA = Array.from({ length: Math.ceil(n / 2) }, (_, i) => i);
                    gB = Array.from({ length: Math.floor(n / 2) }, (_, i) => Math.ceil(n / 2) + i);
                    break;
                }

                cA = centroid(gather(points, gA));
                cB = centroid(gather(points, gB));
            }

            const totalSSE = this.sse(gather(points, gA)) + this.sse(gather(points, gB));
            if (totalSSE < bestSSE) {
                bestSSE = totalSSE;
                bestA = [...gA];
                bestB = [...gB];
            }
        }

        return [bestA, bestB];
    }
}
