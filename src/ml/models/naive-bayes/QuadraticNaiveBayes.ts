import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type { PredictionMetadata, QuadraticNaiveBayesParams } from '../../types';
import { assertModelTrained, calculateFullGaussianLogPdf } from '../../utils';
import { BaseNaiveBayes, type BaseNaiveBayesOptions } from '../base/BaseNaiveBayes';
import { Matrix, type MatrixLike } from '../../matrix';
import {
    calculateMean,
    calculateCovarianceMatrix,
    calculateInverseAndDeterminant,
} from '../../utils';

export type QuadraticNaiveBayesOptions = BaseNaiveBayesOptions & {
    regularization?: number;
};

/**
 * Quadratic Naive Bayes (Quadratic Discriminant Analysis) classifier implementation.
 *
 * Unlike standard Naive Bayes which assumes feature independence, Quadratic NB models
 * the full covariance matrix for each class. This allows it to capture correlations
 * between features, making it more flexible but requiring more training data.
 *
 * The decision boundary is quadratic, hence the name.
 */
export class QuadraticNaiveBayes extends BaseNaiveBayes<QuadraticNaiveBayesParams> {
    private regularization: number;

    constructor(options: QuadraticNaiveBayesOptions) {
        super(options);
        this.regularization = options.regularization ?? 1e-9;
    }

    /**
     * Trains the Quadratic Naive Bayes classifier.
     *
     * @param X - Training features tensor of shape [n_samples, n_features]
     * @param y - Training labels tensor of shape [n_samples, 1] (class indices)
     * @returns Promise resolving to the trained model parameters
     */
    async train(X: Tensor2D, y: Tensor2D): Promise<QuadraticNaiveBayesParams> {
        const [numSamples, numFeatures] = X.shape;

        const [XArray, yArray] = await Promise.all([X.array(), y.data()]);

        // Extract unique classes from labels
        const classSet = new Set<number>(yArray);
        const classes = Array.from(classSet).sort((a, b) => a - b);

        // Initialize storage for class statistics
        const classMeans = Matrix.create([classes.length, numFeatures]);
        const classCovariances: MatrixLike[] = Array.from({ length: classes.length }, () =>
            Matrix.create([numFeatures, numFeatures]),
        );
        const classCovariancesInverse: MatrixLike[] = Array.from({ length: classes.length }, () =>
            Matrix.create([numFeatures, numFeatures]),
        );
        const classCovariancesDeterminant = new Float32Array(classes.length);
        const classPriors = new Float32Array(classes.length);

        // Calculate statistics for each class
        for (let clsIndex = 0; clsIndex < classes.length; clsIndex++) {
            await this.trainingController?.handleControlFlow(true);

            if (this.trainingController?.isTrainingStopped) {
                break;
            }

            const cls = classes[clsIndex];

            // Filter samples belonging to this class
            const classIndices: number[] = [];
            for (let i = 0; i < numSamples; i++) {
                if (yArray[i] === cls) {
                    classIndices.push(i);
                }
            }

            const classCount = classIndices.length;
            classPriors[clsIndex] = classCount / numSamples;

            const classMean = calculateMean(XArray, numFeatures, classIndices);
            classMeans.row(clsIndex).set(classMean);

            // Compute covariance matrix
            const classCovariance = calculateCovarianceMatrix(
                XArray,
                classMean,
                numFeatures,
                classIndices,
            );
            classCovariances[clsIndex] = classCovariance;

            // Add regularization to diagonal
            for (let row = 0; row < numFeatures; row++) {
                classCovariance.array[row * numFeatures + row] += this.regularization;
            }

            // The matrix is already regularized dynamically, so we can pass 0 for fallback epsilon
            const { inverse, determinant } = calculateInverseAndDeterminant(classCovariance, 0);

            classCovariancesInverse[clsIndex] = inverse;
            classCovariancesDeterminant[clsIndex] = determinant;
            classCovariances[clsIndex] = classCovariance;

            this.params = {
                type: 'quadratic',
                classes,
                classMeans,
                classCovariances,
                classCovariancesInverse,
                classCovariancesDeterminant,
                classPriors,
            };

            await this.eventEmitter?.emit('callback', {
                threadId: 0,
                iteration: clsIndex,
                params: this.params,
            });
        }

        return this.params!;
    }

    /**
     * Predicts class labels for input features.
     *
     * @param X - Input features tensor of shape [n_samples, n_features]
     * @param params - Optional model parameters (uses trained params if not provided)
     * @returns Tensor of predicted class indices of shape [n_samples, 1]
     */
    predict(X: Tensor2D, params?: QuadraticNaiveBayesParams): Tensor2D {
        const resolvedParams = params ?? this.params;

        assertModelTrained(resolvedParams);

        const samplesArray = X.arraySync();
        const numSamples = samplesArray.length;
        const predictions = new Float32Array(numSamples);

        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sampleFeatures = samplesArray[sampleIndex];
            const classScores = this.calculateDiscriminantScores(sampleFeatures, resolvedParams);
            const predictedClassIndex = this.probabilityToClassIndex(classScores);
            predictions[sampleIndex] = resolvedParams.classes[predictedClassIndex];
        }

        return tensor2d(predictions, [numSamples, 1]);
    }

    /**
     * Predicts class labels and log-probabilities for input features.
     *
     * @param X - Input features tensor of shape [n_samples, n_features]
     * @param params - Optional model parameters (uses trained params if not provided)
     * @returns Object containing predictions and log-probabilities
     */
    predictWithMetadata(X: Tensor2D, params?: QuadraticNaiveBayesParams): PredictionMetadata {
        const resolvedParams = params ?? this.params;

        assertModelTrained(resolvedParams);

        const samplesArray = X.arraySync();
        const numSamples = samplesArray.length;
        const classLogScoresArray: number[][] = [];
        const predictedClassesArray: number[][] = [];

        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sampleFeatures = samplesArray[sampleIndex];
            const classScores = this.calculateDiscriminantScores(sampleFeatures, resolvedParams);
            const predictedClassIndex = this.probabilityToClassIndex(classScores);

            predictedClassesArray.push([resolvedParams.classes[predictedClassIndex]]);
            classLogScoresArray.push([...classScores]);
        }

        const scoresTensor = tensor2d(classLogScoresArray);
        const predictionsTensor = tensor2d(predictedClassesArray);
        return {
            type: 'classification',
            predictions: predictionsTensor,
            probabilities: scoresTensor,
            dispose() {
                predictionsTensor.dispose();
                scoresTensor.dispose();
            },
        };
    }

    /**
     * Calculates discriminant scores for each class for a given sample.
     * Uses the quadratic discriminant function.
     *
     * @param sample - Feature vector for a single sample
     * @param params - Model parameters
     * @returns Array of discriminant scores for each class
     */
    private calculateDiscriminantScores(
        sample: number[],
        params: QuadraticNaiveBayesParams,
    ): Float32Array {
        const scores = new Float32Array(params.classes.length);

        for (let c = 0; c < params.classes.length; c++) {
            const mean = Matrix.from(params.classMeans).row(c);
            const covInv = params.classCovariancesInverse[c];
            const det = params.classCovariancesDeterminant[c];

            scores[c] =
                Math.log(params.classPriors[c]) +
                calculateFullGaussianLogPdf(sample, mean, covInv, det);
        }

        return scores;
    }
}
