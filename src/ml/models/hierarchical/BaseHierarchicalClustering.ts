import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type {
    ClusteringMetadata,
    HierarchicalClusteringParams,
    Model,
    TrainingControl,
    TrainingEventEmitter,
} from '../../types';
import { assertModelTrained, zeros } from '../../utils';
import { getMatrixFromArray } from '../../utils/matrix';

type BaseOptions = {
    numClusters: number;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

/**
 * Abstract base class for hierarchical clustering models (Agglomerative & Divisive).
 */
export abstract class BaseHierarchicalClustering implements Model<HierarchicalClusteringParams> {
    protected numClusters: number;
    protected params: HierarchicalClusteringParams | null = null;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;
    protected iteration = 0;

    protected constructor({ numClusters, eventEmitter, trainingController }: BaseOptions) {
        this.numClusters = numClusters;
        this.eventEmitter = eventEmitter;
        this.trainingController = trainingController;
    }

    abstract train(X: Tensor2D): Promise<HierarchicalClusteringParams>;

    protected abstract assignPoints(X: Tensor2D, params: HierarchicalClusteringParams): number[];

    predict(X: Tensor2D, params?: HierarchicalClusteringParams): Tensor2D {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const assignments = this.assignPoints(X, modelParams);

        return tensor2d(
            assignments.map((l) => [l]),
            [assignments.length, 1],
        );
    }

    predictWithMetadata(X: Tensor2D, params?: HierarchicalClusteringParams): ClusteringMetadata {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const assignments = this.assignPoints(X, modelParams);
        const assignmentsTensor = tensor2d(
            assignments.map((l) => [l]),
            [assignments.length, 1],
        );

        return {
            type: 'clustering',
            assignments: assignmentsTensor,
            dispose() {
                assignmentsTensor.dispose();
            },
        };
    }

    dispose(): void {
        this.params = null;
    }

    protected buildParams(
        centroidsArr: number[][],
        assignments: Int32Array,
    ): HierarchicalClusteringParams {
        return {
            centroids: getMatrixFromArray(centroidsArr),
            assignments,
        };
    }

    /**
     * Computes cluster centroids by averaging the feature vectors of all points
     * belonging to each cluster.
     *
     * @param XArray - Feature matrix [n_samples, n_features].
     * @param assignments - Per-point labels in the contiguous [0, k-1] range.
     * @param k - Number of clusters.
     */
    protected computeCentroids(XArray: number[][], assignments: Int32Array, k: number): number[][] {
        const n = XArray.length;
        const numFeatures = XArray[0].length;
        const centroids = zeros([k, numFeatures]);
        const clusterSizes = zeros([k]);

        for (let i = 0; i < n; i++) {
            const label = assignments[i];
            clusterSizes[label]++;
            for (let f = 0; f < numFeatures; f++) {
                centroids[label][f] += XArray[i][f];
            }
        }

        for (let c = 0; c < k; c++) {
            if (clusterSizes[c] > 0) {
                for (let f = 0; f < numFeatures; f++) {
                    centroids[c][f] /= clusterSizes[c];
                }
            }
        }

        return centroids;
    }

    /**
     * Remaps sparse cluster IDs in `assignments` to a contiguous [0, k-1] range
     * in-place, using the iteration order of `activeClusters` as the new indices.
     */
    protected relabelClusters(assignments: Int32Array, activeClusters: Set<number>): void {
        const clusterIds = Array.from(activeClusters);
        const idMap = new Map<number, number>();
        clusterIds.forEach((id, idx) => idMap.set(id, idx));
        for (let i = 0; i < assignments.length; i++) {
            assignments[i] = idMap.get(assignments[i])!;
        }
    }

    /**
     * Computes a snapshot of `TParams` from the current cluster map and raw
     * (pre-relabel) assignments so live callbacks can render interim centroids.
     *
     * Cluster IDs are remapped to a contiguous [0, k-1] range in the returned
     * params so they are valid for use with `assignPoints`.
     */
    protected computeInterimParams(
        XArray: number[][],
        clusters: Map<number, number[]>,
        rawAssignments: Int32Array,
    ): HierarchicalClusteringParams {
        const clusterIds = Array.from(clusters.keys());
        const k = clusterIds.length;
        const idMap = new Map<number, number>();
        clusterIds.forEach((id, idx) => idMap.set(id, idx));

        const remappedAssignments = new Int32Array(rawAssignments.length);
        for (let i = 0; i < rawAssignments.length; i++) {
            remappedAssignments[i] = idMap.get(rawAssignments[i])!;
        }

        const centroidsArr = this.computeCentroids(XArray, remappedAssignments, k);
        return {
            centroids: getMatrixFromArray(centroidsArr),
            assignments: remappedAssignments,
        };
    }

    /**
     * Emits a training-progress callback with the current cluster state.
     *
     * @param XArray - Feature matrix as a 2D array.
     * @param clusters - Current cluster map (id → point indices).
     * @param assignments - Raw (pre-relabel) per-point cluster assignments.
     * @param numClusters - Current number of active clusters.
     * @param iteration - Current training iteration index.
     */
    protected async emitCallback(
        XArray: number[][],
        clusters: Map<number, number[]>,
        assignments: Int32Array,
        numClusters: number,
        iteration: number,
    ): Promise<void> {
        if (this.eventEmitter) {
            const params = this.computeInterimParams(XArray, clusters, assignments);

            await this.eventEmitter.emit('callback', {
                threadId: 0,
                assignments,
                iteration,
                numClusters,
                params,
            });
        }
    }
}
