import { type Tensor2D } from '@tensorflow/tfjs';
import type { KNNParams, Model, PredictionMetadata, TrainingEventEmitter } from '../../types';
import { assert } from '../../utils';
import { euclideanDistance, type DistanceMetric } from '../../distance';
import { EPSILON } from '../../constants';

type BaseKNNOptions = {
    k: number;
    distanceMetric?: DistanceMetric;
    weights?: 'uniform' | 'distance';
    eventEmitter?: TrainingEventEmitter;
};

type KNNNeighborBatch = {
    k: number;
    numTestSamples: number;
    neighborValues: Tensor2D;
    neighborDistanceWeights: Tensor2D;
};

export abstract class BaseKNN implements Model<KNNParams> {
    protected readonly k: number;
    protected readonly distanceMetric: DistanceMetric;
    protected readonly weights: 'uniform' | 'distance';
    protected readonly eventEmitter?: TrainingEventEmitter;

    protected params: KNNParams | null = null;

    constructor(options: BaseKNNOptions) {
        assert(Number.isInteger(options.k) && options.k >= 1, 'k must be a positive integer.');

        this.k = options.k;
        this.distanceMetric = options.distanceMetric ?? euclideanDistance;
        this.weights = options.weights ?? 'uniform';
        this.eventEmitter = options.eventEmitter;
    }

    abstract train(X: Tensor2D, y: Tensor2D): Promise<KNNParams>;
    abstract predict(X: Tensor2D, params?: KNNParams): Tensor2D;
    abstract predictWithMetadata(X: Tensor2D, params?: KNNParams): PredictionMetadata;

    dispose(): void {
        this.params?.XTrain.dispose();
        this.params?.yTrain.dispose();

        this.params = null;
    }

    protected async storeTrainingData(
        X: Tensor2D,
        y: Tensor2D,
        classes: number[] = [],
    ): Promise<KNNParams> {
        const [numSamples] = X.shape;
        assert(
            this.k <= numSamples,
            `k (${this.k}) cannot exceed the number of training samples (${numSamples}).`,
        );

        this.disposeStoredParams();

        this.params = {
            type: 'knn',
            XTrain: X.clone(),
            yTrain: y.clone(),
            classes,
        };

        await this.eventEmitter?.emit('callback', {
            threadId: 0,
            iteration: 0,
            params: this.params,
        });

        return this.params;
    }

    protected getNeighborBatch(X: Tensor2D, params: KNNParams): KNNNeighborBatch {
        const { XTrain, yTrain } = params;
        const [numTrainSamples] = XTrain.shape;
        const [numTestSamples] = X.shape;

        const distMatrix = this.distanceMetric(X, XTrain);
        const k = Math.min(this.k, numTrainSamples);

        const { indices: kIndices, values: kNegDists } = distMatrix.neg().topk(k);

        const yFlat = yTrain.reshape([-1]);
        const neighborValuesFlat = yFlat.gather(kIndices.flatten());

        const neighborValues = neighborValuesFlat.reshape([numTestSamples, k]) as Tensor2D;
        const neighborDistanceWeights = kNegDists.neg().maximum(EPSILON).reciprocal() as Tensor2D;

        return {
            k,
            numTestSamples,
            neighborValues,
            neighborDistanceWeights,
        };
    }

    protected disposeStoredParams(): void {
        this.params?.XTrain.dispose();
        this.params?.yTrain.dispose();
    }
}
