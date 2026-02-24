import { concat, oneHot, tidy, type Scalar, type Tensor1D, type Tensor2D } from '@tensorflow/tfjs';
import type {
    Model,
    ModelRepresentation,
    PredictionMetadata,
    TrainingEventEmitter,
    Scaler,
    ScalerParams,
    ScalerState,
} from '../types';
import type { TransformationFn } from '../data-processing/transformation';

export type FeatureTransformConfig = {
    preScaler?: Scaler<ScalerParams>;
    transformations?: TransformationFn[];
    postScaler?: Scaler<ScalerParams>;
};

export class PipelineModel<T extends ModelRepresentation> implements Model<T> {
    private model: Model<T>;
    private featureTransform?: FeatureTransformConfig;
    private eventEmitter?: TrainingEventEmitter;

    private _cachedProcessedData: Map<number, Tensor2D> = new Map();

    constructor(
        model: Model<T>,
        featureTransform?: FeatureTransformConfig,
        eventEmitter?: TrainingEventEmitter,
    ) {
        this.model = model;
        this.featureTransform = featureTransform;
        this.eventEmitter = eventEmitter;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<T> {
        this.eventEmitter?.emit('state', 'transforming');

        X = this.prepareFeatures(X, true);
        y = this.prepareLabels(y);

        this.eventEmitter?.emit('state', 'training');

        const result = await this.model.train(X, y);

        X.dispose();
        y.dispose();

        return result;
    }

    predict(X: Tensor2D, theta?: T): Tensor2D {
        X = this.prepareFeatures(X);

        const result = this.model.predict(X, theta);

        X.dispose();

        return result;
    }

    predictWithMetadata(X: Tensor2D, theta?: T): PredictionMetadata {
        X = this.prepareFeatures(X);

        const result = this.model.predictWithMetadata(X, theta);

        X.dispose();

        return result;
    }

    evaluate(X: Tensor2D, y: Tensor2D, theta?: T): [Tensor2D, Tensor2D, Scalar] {
        X = this.prepareFeatures(X);
        y = this.prepareLabels(y);

        const result = this.model.evaluate(X, y, theta);

        X.dispose();
        y.dispose();

        return result;
    }

    async extractScalerParams(): Promise<ScalerState> {
        const { preScaler, postScaler } = this.featureTransform ?? {};

        const prePromise = preScaler?.extractParameters?.();
        const postPromise = postScaler?.extractParameters?.();

        if (!prePromise && !postPromise) {
            return {};
        }

        const [preScalerParams, postScalerParams] = await Promise.all([prePromise, postPromise]);

        return {
            preScaler: preScalerParams,
            postScaler: postScalerParams,
        };
    }

    dispose(withDependencies?: boolean): void {
        this._cachedProcessedData.forEach((tensor) => tensor?.dispose());
        this._cachedProcessedData.clear();
        this.model.dispose(withDependencies);
    }

    prepareFeatures(features: Tensor2D, isTraining = false): Tensor2D {
        const { preScaler, postScaler, transformations = [] } = this.featureTransform ?? {};

        if (!this._cachedProcessedData.has(features.id)) {
            const processedFeatures = tidy(() => {
                try {
                    if (isTraining && preScaler) {
                        preScaler.fit(features);
                    }
                    const preScaledFeatures = preScaler?.transform(features) ?? features;

                    let processedFeatures = preScaledFeatures.clone();

                    for (const transform of transformations) {
                        const additionalFeatures = transform(preScaledFeatures);

                        if (additionalFeatures !== null) {
                            processedFeatures = concat([processedFeatures, additionalFeatures], 1);
                        }
                    }

                    if (isTraining && postScaler) {
                        postScaler.fit(processedFeatures);
                    }

                    const normalizedFeatures = postScaler
                        ? postScaler.transform(processedFeatures)
                        : processedFeatures;

                    return normalizedFeatures;
                } catch (error) {
                    console.error('Error processing features:', error);
                    throw error;
                }
            });

            this._cachedProcessedData.set(features.id, processedFeatures);
        }

        // Clone to ensure downstream code does not mutate or dispose the original labels tensor
        return this._cachedProcessedData.get(features.id)!.clone();
    }

    prepareLabels(labels: Tensor2D): Tensor2D {
        const usesOneHotLabels = this.model.usesOneHotLabels?.();

        if (!this._cachedProcessedData.has(labels.id)) {
            const processedLabel = tidy(() => {
                let processedLabel;
                if (usesOneHotLabels) {
                    const labelsFlat = labels.squeeze().toInt() as Tensor1D;
                    const numClasses = new Set(labelsFlat.arraySync()).size; // WebGPU does not yet support the unique() function
                    processedLabel = oneHot(labelsFlat, numClasses) as Tensor2D;
                } else {
                    // Clone to ensure the model manages tensor disposal
                    processedLabel = labels.clone();
                }

                return processedLabel;
            });

            this._cachedProcessedData.set(labels.id, processedLabel);
        }

        // Clone to ensure downstream code does not mutate or dispose the original labels tensor
        return this._cachedProcessedData.get(labels.id)!.clone();
    }
}
