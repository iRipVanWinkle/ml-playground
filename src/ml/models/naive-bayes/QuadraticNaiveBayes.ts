import { tidy, tensor2d, type Scalar, type Tensor2D, scalar } from '@tensorflow/tfjs';
import type { QuadraticNaiveBayesParams } from '../../types';
import { assertModelTrained } from '../../utils';
import { EPSILON } from '../../constants';
import { BaseNaiveBayes, type BaseNaiveBayesOptions } from '../base/BaseNaiveBayes';
import { Matrix, type MatrixLike } from '../../matrix';

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

            const classMean = classMeans.row(clsIndex);

            // Sum feature values across all samples in this class
            for (const sampleIdx of classIndices) {
                const sample = XArray[sampleIdx];
                for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                    classMean[featureIdx] += sample[featureIdx];
                }
            }

            // Compute mean by dividing sums by class count
            for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                classMean[featureIdx] /= classCount;
            }

            const classCovariance = classCovariances[clsIndex];

            // Compute covariance matrix: Cov[i,j] = E[(X_i - μ_i)(X_j - μ_j)]
            const centeredData: Float32Array[] = [];
            for (const sampleIdx of classIndices) {
                const deviations = new Float32Array(numFeatures);
                for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                    deviations[featureIdx] = XArray[sampleIdx][featureIdx] - classMean[featureIdx];
                }
                centeredData.push(deviations);
            }

            // Calculate covariance matrix elements using outer product of deviations
            for (let row = 0; row < numFeatures; row++) {
                for (let col = 0; col < numFeatures; col++) {
                    let covarianceSum = 0;
                    for (const deviations of centeredData) {
                        covarianceSum += deviations[row] * deviations[col];
                    }

                    // Normalize by sample count and add regularization to diagonal
                    classCovariance.array[row * numFeatures + col] = covarianceSum / classCount;
                    if (row === col) {
                        classCovariance.array[row * numFeatures + col] += this.regularization;
                    }
                }
            }

            classCovariancesInverse[clsIndex] = this.invertMatrix(classCovariance);
            classCovariancesDeterminant[clsIndex] = this.determinant(classCovariance);
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
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const XArray = X.arraySync();
        const numSamples = XArray.length;
        const predictions = new Float32Array(numSamples);

        for (let i = 0; i < numSamples; i++) {
            const sample = XArray[i];
            const scores = this.calculateDiscriminantScores(sample, modelParams);
            const maxIdx = scores.indexOf(Math.max(...scores));
            predictions[i] = modelParams.classes[maxIdx];
        }

        return tensor2d(predictions, [numSamples, 1]);
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

    /**
     * Inverts a positive-definite symmetric matrix using Cholesky decomposition.
     * Falls back to regularization if the matrix is not positive definite.
     *
     * @complexity O(n³) where n is matrix dimension
     * @param matrix - Square symmetric positive-definite matrix
     * @returns Inverted matrix
     */
    private invertMatrix(matrix: MatrixLike): MatrixLike {
        const n = matrix.shape[0];
        const array = matrix.array;

        // Cholesky decomposition: A = L * L^T
        const L = Matrix.create([n, n]).array;

        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let sum = 0;
                for (let k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }

                if (i === j) {
                    const val = array[i * n + i] - sum;
                    if (val <= 0) {
                        // Not positive definite, add regularization
                        L[i * n + j] = Math.sqrt(Math.abs(val) + this.regularization);
                    } else {
                        L[i * n + j] = Math.sqrt(val);
                    }
                } else {
                    L[i * n + j] = (array[i * n + j] - sum) / L[j * n + j];
                }
            }
        }

        // Invert L (lower triangular)
        const LInv = Matrix.create([n, n]).array;
        for (let i = 0; i < n; i++) {
            LInv[i * n + i] = 1 / L[i * n + i];
            for (let j = i - 1; j >= 0; j--) {
                let sum = 0;
                for (let k = j + 1; k <= i; k++) {
                    sum += L[i * n + k] * LInv[k * n + j];
                }
                LInv[i * n + j] = -sum / L[i * n + i];
            }
        }

        // A^-1 = (L^T)^-1 * L^-1
        const inverse = Matrix.create([n, n]);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                for (let k = Math.max(i, j); k < n; k++) {
                    inverse.array[i * n + j] += LInv[k * n + i] * LInv[k * n + j];
                }
            }
        }

        return inverse;
    }

    /**
     * Calculates the determinant of a square matrix using LU decomposition.
     *
     * @param matrix - Square matrix
     * @returns Determinant value
     */
    private determinant(matrix: MatrixLike): number {
        const n = matrix.shape[0];
        const m = matrix.array.slice(); // Copy matrix

        let det = 1;

        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(m[k * n + i]) > Math.abs(m[maxRow * n + i])) {
                    maxRow = k;
                }
            }

            if (maxRow !== i) {
                [m[i], m[maxRow]] = [m[maxRow], m[i]];
                det *= -1;
            }

            if (Math.abs(m[i * n + i]) < EPSILON) {
                return 0; // Singular matrix
            }

            det *= m[i * n + i];
            // Eliminate below
            for (let k = i + 1; k < n; k++) {
                const factor = m[k * n + i] / m[i * n + i];
                for (let j = i; j < n; j++) {
                    m[k * n + j] -= factor * m[i * n + j];
                }
            }
        }

        return det;
    }
}
