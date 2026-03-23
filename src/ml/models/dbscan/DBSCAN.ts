import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type {
    DBSCANParams,
    ClusteringMetadata,
    TrainingControl,
    TrainingEventEmitter,
    Model,
} from '../../types';
import { getMatrixFromArray } from '../../matrix';
import { assert, assertModelTrained } from '../../utils';
import { euclideanDistance, type DistanceMetric } from '../../distance';

const NOISE_LABEL = -1;
const UNVISITED = -2;

export type ModelOptions = {
    epsilon: number;
    minPoints: number;
    distanceMetric?: DistanceMetric;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

type Context = {
    XArray: number[][];
    labels: Int32Array;
    isCoreMap: Map<number, number>;
    neighbors: number[][];
    numClusters: number;
};

export class DBSCAN implements Model<DBSCANParams> {
    private epsilon: number;
    private minPoints: number;
    private distanceMetric: DistanceMetric;
    private eventEmitter?: TrainingEventEmitter;
    private trainingController?: TrainingControl;

    private params: DBSCANParams | null = null;
    private iteration = 0;

    constructor(options: ModelOptions) {
        assert(options.epsilon > 0, 'Epsilon must be a positive number.');
        assert(
            Number.isInteger(options.minPoints) && options.minPoints >= 1,
            'minPoints must be a positive integer.',
        );

        this.epsilon = options.epsilon;
        this.minPoints = options.minPoints;
        this.distanceMetric = options.distanceMetric ?? euclideanDistance;
        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;
    }

    async train(X: Tensor2D): Promise<DBSCANParams> {
        const XArray = X.arraySync();
        const numSamples = XArray.length;

        this.iteration = 0;

        // Labels: UNVISITED → not yet processed, NOISE_LABEL → noise, >= 0 → cluster id
        const labels = new Int32Array(numSamples).fill(UNVISITED);

        // Step 1: Find neighbors for each point
        const neighbors = this.computeNeighborLists(X);

        // Step 2: Identify core points (points with enough neighbors)
        const isCoreMap = new Map<number, number>();
        const corePoints = [];
        for (let i = 0; i < numSamples; i++) {
            if (neighbors[i].length >= this.minPoints) {
                isCoreMap.set(i, corePoints.length); // map original index to core point index
                corePoints.push(XArray[i]);
            }
        }

        this.params = {
            type: 'dbscan',
            corePoints: getMatrixFromArray(corePoints),
            coreLabels: new Int32Array(corePoints.length).fill(UNVISITED),
        };

        const context = {
            XArray,
            labels,
            isCoreMap,
            neighbors,
            numClusters: 0,
        };

        // Step 3: Process each point to form clusters
        let clusterId = 0;

        for (let pointIndex = 0; pointIndex < numSamples; pointIndex++) {
            if (this.trainingController?.isTrainingStopped) {
                break;
            }

            if (labels[pointIndex] !== UNVISITED || !isCoreMap.has(pointIndex)) {
                continue;
            }

            context.numClusters++;

            // Core point → seed a new cluster
            await this.updateLabel(pointIndex, clusterId, context);

            // Expand this cluster point by point
            await this.expandCluster(pointIndex, clusterId, context);

            clusterId++;
        }

        // Mark any remaining unvisited points as noise
        for (let i = 0; i < numSamples; i++) {
            if (labels[i] === UNVISITED) {
                labels[i] = NOISE_LABEL;
            }
        }

        // Final callback with the complete state
        await this.emitCallback(context);

        return this.params;
    }

    /**
     * Predicts cluster assignments for new data points.
     *
     * Each point is assigned to the cluster of the nearest core point
     * if the distance is within `epsilon`, otherwise marked as noise (-1).
     *
     * @param X - Input features tensor of shape [n_samples, n_features]
     * @param params - Optional model parameters (uses trained params if not provided)
     * @returns Tensor of shape [n_samples, 1] with cluster labels (-1 for noise)
     */
    predict(X: Tensor2D, params?: DBSCANParams): Tensor2D {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const assignments = this.assignPoints(X, modelParams);

        return tensor2d(
            assignments.map((l) => [l]),
            [assignments.length, 1],
        );
    }

    /**
     * Predicts cluster assignments with clustering metadata.
     */
    predictWithMetadata(X: Tensor2D, params?: DBSCANParams): ClusteringMetadata {
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

    /**
     * Precomputes the neighbor list for every point.
     * neighbors[i] contains the indices of all points within epsilon of point i.
     */
    private computeNeighborLists(data: Tensor2D): number[][] {
        const distMatrix = this.distanceMetric(data, data);
        const distArray = distMatrix.arraySync();
        distMatrix.dispose();

        const n = distArray.length;
        const neighbors: number[][] = Array.from({ length: n }, () => []);

        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (distArray[i][j] <= this.epsilon) {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }

            // Include self in neighbors for core point counting
            neighbors[i].push(i);
        }

        return neighbors;
    }

    /**
     * Expands a cluster from a seed core point using breadth-first traversal.
     * Emits a callback for each point added to the cluster.
     *
     * Density-reachability: a point q is density-reachable from p if there
     * exists a chain of core points p → p1 → … → pn → q where each step
     * is within epsilon distance.
     */
    private async expandCluster(seedIndex: number, clusterId: number, context: Context) {
        const { labels, isCoreMap, neighbors } = context;
        const queue = [...neighbors[seedIndex]];
        let head = 0;

        while (head < queue.length) {
            if (this.trainingController?.isTrainingStopped) break;

            const pointIndex = queue[head++];

            // Skip already-assigned points without consuming a step
            if (labels[pointIndex] !== UNVISITED && labels[pointIndex] !== NOISE_LABEL) {
                continue;
            }

            await this.trainingController?.handleControlFlow(true);

            // Noise points become border points of this cluster
            if (labels[pointIndex] === NOISE_LABEL) {
                await this.updateLabel(pointIndex, clusterId, context);
                continue;
            }

            // Unvisited points become part of this cluster
            await this.updateLabel(pointIndex, clusterId, context);

            // If this neighbor is also a core point, add its neighbors to the queue
            if (isCoreMap.has(pointIndex)) {
                for (const neighbor of neighbors[pointIndex]) {
                    if (labels[neighbor] === UNVISITED || labels[neighbor] === NOISE_LABEL) {
                        queue.push(neighbor);
                    }
                }
            }
        }
    }

    private async updateLabel(
        pointIndex: number,
        newLabel: number,
        context: Context,
    ): Promise<void> {
        const { labels, isCoreMap } = context;
        labels[pointIndex] = newLabel;

        if (this.params && isCoreMap.has(pointIndex)) {
            // update coreLabels if this point is a core point
            this.params.coreLabels[isCoreMap.get(pointIndex)!] = newLabel;
        }

        await this.emitCallback(context, pointIndex);
    }

    private async emitCallback(context: Context, activePointIndex?: number): Promise<void> {
        if (!this.eventEmitter || !this.params) return;

        const { labels, numClusters } = context;

        await this.eventEmitter.emit('callback', {
            threadId: 0,
            iteration: this.iteration++,
            assignments: labels,
            numClusters,
            activePointIndex,
            epsilon: this.epsilon,
            params: this.params,
        });
    }

    /**
     * Assigns new points to the nearest core point's cluster, or noise.
     */
    private assignPoints(X: Tensor2D, params: DBSCANParams): number[] {
        const { corePoints, coreLabels } = params;
        const numSamples = X.shape[0];

        if (corePoints.shape[0] === 0) {
            return new Array(numSamples).fill(NOISE_LABEL);
        }

        // Compute full [numSamples, numCorePoints] distance matrix on GPU
        const corePointsTensor = tensor2d(corePoints.array, corePoints.shape);
        const distMatrix = this.distanceMetric(X, corePointsTensor);
        corePointsTensor.dispose();

        const minDists = distMatrix.min(1);
        const bestIndices = distMatrix.argMin(1);
        distMatrix.dispose();

        const minDistsData = minDists.dataSync();
        const bestIndicesData = bestIndices.dataSync();
        minDists.dispose();
        bestIndices.dispose();

        const assignments: number[] = new Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
            assignments[i] =
                minDistsData[i] <= this.epsilon ? coreLabels[bestIndicesData[i]] : NOISE_LABEL;
        }

        return assignments;
    }
}
