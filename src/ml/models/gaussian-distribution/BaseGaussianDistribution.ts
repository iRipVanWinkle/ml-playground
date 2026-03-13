import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type {
    GaussianDistributionParams,
    TrainingControl,
    TrainingEventEmitter,
    Model,
    AnomalyDetectionMetadata,
} from '../../types';
import { assertModelTrained } from '../../utils';

export type BaseGaussianDistributionOptions = {
    threshold?: number;
    varianceSmoothing?: number;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

/**
 * Abstract base class for Gaussian Distribution anomaly detection models.
 *
 * Provides shared predict/predictWithMetadata logic. Subclasses implement
 * specific covariance strategies (diagonal or full).
 */
export abstract class BaseGaussianDistribution<T extends GaussianDistributionParams>
    implements Model<T>
{
    protected threshold: number;
    protected varianceSmoothing: number;
    protected params: T | null = null;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;

    constructor(options: BaseGaussianDistributionOptions) {
        this.threshold = options.threshold ?? 0.01;
        this.varianceSmoothing = options.varianceSmoothing ?? 1e-9;
        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;
    }

    abstract train(X: Tensor2D): Promise<T>;

    /**
     * Predicts anomaly scores for input features.
     * Returns 1 for anomalies (low probability), 0 for normal (high probability).
     */
    predict(X: Tensor2D, params?: T): Tensor2D {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const XArray = X.arraySync();
        const numSamples = XArray.length;
        const predictions: number[][] = [];

        for (let i = 0; i < numSamples; i++) {
            const sample = XArray[i];
            const probability = this.calculateProbability(sample, modelParams);
            predictions.push([probability < modelParams.threshold ? 1 : 0]);
        }

        return tensor2d(predictions);
    }

    /**
     * Predicts anomaly scores with metadata for input features.
     * Returns predictions (1 for anomalies, 0 for normal) and probability densities.
     */
    predictWithMetadata(X: Tensor2D, params?: T): AnomalyDetectionMetadata {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const XArray = X.arraySync();
        const numSamples = XArray.length;
        const predictionsArray: number[][] = [];
        const probabilitiesArray: number[][] = [];

        for (let i = 0; i < numSamples; i++) {
            const sample = XArray[i];
            const probability = this.calculateProbability(sample, modelParams);
            predictionsArray.push([probability < modelParams.threshold ? 1 : 0]);
            probabilitiesArray.push([probability]);
        }

        const predictionsTensor = tensor2d(predictionsArray);
        const probabilitiesTensor = tensor2d(probabilitiesArray);

        return {
            type: 'anomaly-detection',
            predictions: predictionsTensor,
            probabilities: probabilitiesTensor,
            dispose() {
                predictionsTensor.dispose();
                probabilitiesTensor.dispose();
            },
        };
    }

    /**
     * Disposes of any resources used by the model.
     */
    dispose(): void {
        this.params = null;
    }

    /**
     * Calculates the probability density for a single sample.
     * Implemented by subclasses.
     */
    protected abstract calculateProbability(sample: number[], params: T): number;
}
