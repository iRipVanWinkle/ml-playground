import { tidy, tensor2d, type Scalar, type Tensor2D, scalar } from '@tensorflow/tfjs';
import type { PredictionMetadata, QuadraticNaiveBayesParams } from '../../types';
import { assertModelTrained } from '../../utils';
import { EPSILON } from '../../constants';
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
     * Evaluates the model on test data.
     *
     * @param X - Test features tensor of shape [n_samples, n_features]
     * @param y - True labels tensor of shape [n_samples, 1]
     * @param params - Optional model parameters (uses trained params if not provided)
     * @returns Tuple of [predictions, probabilities, loss]
     */
    evaluate(
        X: Tensor2D,
        y: Tensor2D,
        params?: QuadraticNaiveBayesParams,
    ): [Tensor2D, Tensor2D, Scalar] {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const XArray = X.arraySync();
        const yArray = y.dataSync(); // get the data as a flat array
        const numSamples = XArray.length;

        const predictions: number[][] = [];
        const probabilities: number[][] = [];
        let correctCount = 0;

        for (let i = 0; i < numSamples; i++) {
            const sample = XArray[i];
            const scores = this.calculateDiscriminantScores(sample, modelParams);

            // Convert scores to probabilities using softmax
            const maxScore = Math.max(...scores);
            const expScores = scores.map((s) => Math.exp(s - maxScore));
            const sumExpScores = expScores.reduce((a, b) => a + b, 0);
            const probs = expScores.map((s) => s / sumExpScores);

            const maxIdx = scores.indexOf(maxScore);
            const predictedClass = modelParams.classes[maxIdx];

            predictions.push([predictedClass]);
            probabilities.push(Array.from(probs));
            if (predictedClass === yArray[i]) {
                correctCount++;
            }
        }

        const result = tidy(() => {
            const yPred = tensor2d(predictions);
            const yProbs = tensor2d(probabilities);

            // Loss is negative log likelihood (accuracy-based approximation)
            const accuracy = correctCount / numSamples;
            const loss = scalar(-Math.log(accuracy + EPSILON));

            return [yPred, yProbs, loss] as [Tensor2D, Tensor2D, Scalar];
        });

        return result;
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
        const numFeatures = sample.length;
        const scores = new Float32Array(params.classes.length);

        for (let c = 0; c < params.classes.length; c++) {
            const mean = Matrix.from(params.classMeans).row(c);
            const covInv = params.classCovariancesInverse[c].array;
            const det = params.classCovariancesDeterminant[c];

            // Calculate (x - mean)
            const diff = new Float32Array(numFeatures);
            for (let j = 0; j < diff.length; j++) {
                diff[j] = sample[j] - mean[j];
            }

            // Quadratic discriminant score:
            // score = log(P(class)) - 0.5 * log(|Cov|) - 0.5 * (x-mean)^T * Cov^-1 * (x-mean)
            let mahalanobis = 0;
            for (let j = 0; j < numFeatures; j++) {
                const rowOffset = j * numFeatures;
                for (let k = 0; k < numFeatures; k++) {
                    mahalanobis += diff[j] * covInv[rowOffset + k] * diff[k];
                }
            }

            const score =
                Math.log(params.classPriors[c]) -
                0.5 * Math.log(Math.abs(det) + EPSILON) -
                0.5 * mahalanobis;

            scores[c] = score;
        }

        return scores;
    }
}
