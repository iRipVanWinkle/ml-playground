import { concat, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs-core/dist/base';
import type { Model, TrainingEventEmitter } from './types';
import type { NormalizatorFn } from './data-processing/normalization';
import type { TransformationFn } from './data-processing/transformation';

export type FeatureTransformConfig = {
    polynomialDegree?: number;
    sinusoidDegree?: number;
    normalizeFunction?: NormalizatorFn;
    transformations?: TransformationFn[];
};

export class ModelPipeline implements Model {
    private model: Model;
    private featureTransform?: FeatureTransformConfig;
    private eventEmitter?: TrainingEventEmitter;

    private _cachedProcessedData: Map<number, Tensor2D> = new Map();

    constructor(
        model: Model,
        featureTransform?: FeatureTransformConfig,
        eventEmitter?: TrainingEventEmitter,
    ) {
        this.model = model;
        this.featureTransform = featureTransform;
        this.eventEmitter = eventEmitter;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<unknown> {
        this.eventEmitter?.emit('state', 'transforming');

        const usesOneHotLabels = this.model.usesOneHotLabels?.() ? 'one-hot' : undefined;
        X = this.prepareFeatures(X);
        y = this.prepareLabels(y, usesOneHotLabels);

        this.eventEmitter?.emit('state', 'training');

        const result = await this.model.train(X, y);

        X.dispose();
        y.dispose();

        return result;
    }

    predict(X: Tensor2D, theta?: unknown): Tensor2D {
        X = this.prepareFeatures(X);

        const result = this.model.predict(X, theta);

        X.dispose();

        return result;
    }

    evaluate(X: Tensor2D, y: Tensor2D, theta?: unknown): [Tensor2D, Tensor2D, Scalar] {
        const usesOneHotLabels = this.model.usesOneHotLabels?.() ? 'one-hot' : undefined;
        X = this.prepareFeatures(X);
        y = this.prepareLabels(y, usesOneHotLabels);

        const result = this.model.evaluate?.(X, y, theta);

        X.dispose();
        y.dispose();

        return result;
    }

    dispose(withDependencies?: boolean): void {
        this._cachedProcessedData.forEach((tensor) => tensor?.dispose());
        this._cachedProcessedData.clear();
        this.model.dispose(withDependencies);
    }

    stop(): void {
        this.model.stop();
        this.eventEmitter?.emit('state', 'stopped');
    }

    pause(): void {
        this.model.pause();
        this.eventEmitter?.emit('state', 'paused');
    }

    resume(): void {
        this.model.resume();
        this.eventEmitter?.emit('state', 'training');
    }

    step(): void {
        this.model.step();
        this.eventEmitter?.emit('state', 'stepped-forward');
    }

    prepareFeatures(features: Tensor2D): Tensor2D {
        const options = this.featureTransform;
        const transformations = options?.transformations ?? [];
        const normalizeFunction = options?.normalizeFunction ?? ((x) => x);

        if (!this._cachedProcessedData.has(features.id)) {
            const processedFeatures = tidy(() => {
                // Normalize the data using the normalize function
                const normalizedFeatures = normalizeFunction(features.clone());
                let processedFeatures = normalizedFeatures;

                for (const transform of transformations) {
                    const additionalData = transform(normalizedFeatures);

                    if (additionalData !== null) {
                        processedFeatures = concat([processedFeatures, additionalData], 1);
                    }
                }

                return processedFeatures;
            });

            this._cachedProcessedData.set(features.id, processedFeatures);
        }

        return this._cachedProcessedData.get(features.id)!.clone() as Tensor2D;
    }

    prepareLabels(labels: Tensor2D, convert?: 'one-hot'): Tensor2D {
        if (!this._cachedProcessedData.has(labels.id)) {
            const processedLabel = tidy(() => {
                let processedLabel;
                if (convert === 'one-hot') {
                    const numClasses = labels.unique().values.shape[0];
                    processedLabel = labels.flatten().toInt().oneHot(numClasses, 1, 0) as Tensor2D;
                } else {
                    processedLabel = labels.clone();
                }

                return processedLabel;
            });

            this._cachedProcessedData.set(labels.id, processedLabel);
        }

        return this._cachedProcessedData.get(labels.id)!.clone() as Tensor2D;
    }
}
