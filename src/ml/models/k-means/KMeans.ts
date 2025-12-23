import {
    type Tensor2D,
    type Scalar,
    tidy,
    oneHot,
    variable,
    Variable,
    Rank,
} from '@tensorflow/tfjs';
import type { Model, TrainingControl, TrainingEventEmitter } from '../../types';
import { EPSILON } from '../../constants';
import { assert, assertModelTrained } from '../../utils';
import { euclideanDistance } from '../../distance';
import { centroidInitializationFactory } from '../../factories';

type InitializeCentroids = (X: Tensor2D, k: number) => Tensor2D;
type DistanceMetric = (X: Tensor2D, centroid: Tensor2D) => Tensor2D;

export type ModelOptions = {
    numClusters: number;
    maxIterations: number;
    tolerance?: number;
    distanceMetric?: DistanceMetric;
    initializeCentroids?: InitializeCentroids;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

export class KMeans implements Model<Tensor2D> {
    protected numClusters: number;
    protected maxIterations: number;
    protected tolerance?: number;
    protected distanceMetric: DistanceMetric;
    protected initializeCentroids: InitializeCentroids;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;

    protected centroids?: Variable<Rank.R2>;

    constructor(options: ModelOptions) {
        assert(options.numClusters >= 2, 'Number of clusters (k) must be at least 2.');

        this.numClusters = options.numClusters;
        this.maxIterations = options.maxIterations;
        this.tolerance = options.tolerance;
        this.initializeCentroids =
            options.initializeCentroids ?? centroidInitializationFactory({ type: 'random' });
        this.distanceMetric = options.distanceMetric ?? euclideanDistance;

        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;
    }

    async train(X: Tensor2D): Promise<Tensor2D> {
        const [numSamples] = X.shape;

        assert(
            this.numClusters <= numSamples,
            `Number of clusters (${this.numClusters}) cannot exceed number of samples (${numSamples})`,
        );

        const centroidInit = this.initializeCentroids(X, this.numClusters);
        this.centroids = variable(centroidInit);
        let prevInertia = Infinity;

        centroidInit.dispose();

        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            // Handle pause/step logic
            await this.trainingController?.handleControlFlow();

            if (this.trainingController?.isTrainingStopped) {
                break;
            }

            const assignments = this.findClosestCentroids(X, this.centroids!);
            const centroids = this.computeCentroids(X, assignments);

            // Inertia isn't required by K-Means algorithm but computed here
            // to measure convergence improvement and enable early stopping via tolerance
            const inertia = this.computeInertia(X, centroids, assignments);

            this.centroids.assign(centroids);

            const inertiaValue = (await inertia.data())[0];
            await this.eventEmitter?.emit('callback', {
                threadId: 0,
                centroids: this.centroids,
                iteration,
                assignments,
                inertia: inertiaValue,
            });

            centroids.dispose();
            inertia.dispose();
            assignments.dispose();

            if (this.checkEarlyStopping(prevInertia, inertiaValue, iteration)) {
                break;
            }

            prevInertia = inertiaValue;
        }

        return this.centroids;
    }

    predict(X: Tensor2D, centroids?: Tensor2D | undefined): Tensor2D {
        assertModelTrained(centroids ?? this.centroids);

        const usedCentroids = centroids ?? this.centroids!;
        const assignments = this.findClosestCentroids(X, usedCentroids);
        return assignments.expandDims(1);
    }

    evaluate(
        X: Tensor2D,
        _: Tensor2D,
        centroids?: Tensor2D | undefined,
    ): [Tensor2D, Tensor2D, Scalar] {
        assertModelTrained(centroids ?? this.centroids);

        const usedCentroids = centroids ?? this.centroids!;
        const assignments = this.findClosestCentroids(X, usedCentroids);
        return [X, assignments, this.computeInertia(X, usedCentroids, assignments)];
    }

    dispose(): void {
        this.centroids?.dispose();
    }

    private findClosestCentroids(data: Tensor2D, centroids: Tensor2D): Tensor2D {
        return tidy(() => {
            const distances = this.distanceMetric(data, centroids);

            // Argmin over centroids
            return distances.argMin(1).expandDims(1);
        });
    }

    private computeCentroids(data: Tensor2D, assignments: Tensor2D): Tensor2D {
        return tidy(() => {
            const labels = assignments.squeeze();
            const hot = oneHot(labels, this.numClusters);
            const counts = hot.sum(0).expandDims(1);
            const summed = hot.transpose().matMul(data);

            return summed.div(counts.add(EPSILON));
        });
    }

    private computeInertia(data: Tensor2D, centroids: Tensor2D, assignments: Tensor2D): Scalar {
        return tidy(() => {
            const assignedCentroids = centroids.gather(assignments.squeeze());

            const squaredDistances = data.sub(assignedCentroids).square().sum(1);

            // Total inertia
            return squaredDistances.sum();
        });
    }

    private checkEarlyStopping(
        prevInertia: number,
        currentInertia: number,
        iteration: number,
    ): boolean {
        if (this.tolerance === undefined) {
            return false;
        }

        if (!Number.isFinite(prevInertia)) {
            return false;
        }

        const improvement = Math.abs(prevInertia - currentInertia) / prevInertia;
        if (improvement < this.tolerance) {
            this.eventEmitter?.emit(
                'info',
                `Early stopping at iteration ${iteration + 1}, improvement: ${improvement}`,
            );
            return true;
        }

        return false;
    }
}
