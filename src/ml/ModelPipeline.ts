import { concat, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs-core/dist/base';
import type { Model } from './types';
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

    protected featureTransform?: FeatureTransformConfig;

    private _cachedProcessedData: Map<number, Tensor2D> = new Map();

    constructor(model: Model, featureTransform?: FeatureTransformConfig) {
        this.model = model;
        this.featureTransform = featureTransform;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<unknown> {
        const usesOneHotLabels = this.model.usesOneHotLabels?.() ? 'one-hot' : undefined;

        X = this.prepareFeatures(X);
        y = this.prepareLabels(y, usesOneHotLabels);

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
        X = this.prepareFeatures(X);
        y = this.prepareLabels(y);

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
    }

    pause(): void {
        this.model.pause();
    }

    resume(): void {
        this.model.resume();
    }

    step(): void {
        this.model.step();
    }

    private prepareFeatures(features: Tensor2D): Tensor2D {
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

    private prepareLabels(labels: Tensor2D, convert?: 'one-hot'): Tensor2D {
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
