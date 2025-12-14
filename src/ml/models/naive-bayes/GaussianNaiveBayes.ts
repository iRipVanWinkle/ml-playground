import { tidy, tensor2d, type Scalar, type Tensor2D, scalar } from '@tensorflow/tfjs';
import type { GaussianNaiveBayesParams } from '../../types';
import { assertModelTrained } from '../../utils';
import { BaseNaiveBayes, type BaseNaiveBayesOptions } from '../base/BaseNaiveBayes';
import { Matrix } from '../../matrix';
import { EPSILON } from '../../constants';

export type GaussianNaiveBayesOptions = BaseNaiveBayesOptions & {
    varianceSmoothing?: number;
};

/**
 * Gaussian Naive Bayes classifier implementation.
 *
 * Uses the Naive Bayes algorithm with Gaussian (normal) distribution assumption
 * for continuous features. Suitable for classification tasks with continuous data.
 */
export class GaussianNaiveBayes extends BaseNaiveBayes<GaussianNaiveBayesParams> {
    private varianceSmoothing: number;

    constructor(options: GaussianNaiveBayesOptions) {
        super(options);
        this.varianceSmoothing = options.varianceSmoothing ?? 1e-9;
    }

    /**
     * Trains the Gaussian Naive Bayes classifier.
     *
     * @param X - Training features tensor of shape [n_samples, n_features]
     * @param y - Training labels tensor of shape [n_samples, 1] (class indices)
     * @returns Promise resolving to the trained model parameters
     */
    async train(X: Tensor2D, y: Tensor2D): Promise<GaussianNaiveBayesParams> {
        const [numSamples, numFeatures] = X.shape;

        const [XArray, yArray] = await Promise.all([
            X.array(),
            y.data(), // get the data as a flat array
        ]);

        // Extract unique classes from labels
        const classSet = new Set<number>(yArray);
        const classes = Array.from(classSet).sort((a, b) => a - b);
        const numClasses = classes.length;

        // Initialize default values for all classes
        const classMeans = Matrix.create([numClasses, numFeatures]);
        const classVariances = Matrix.create([numClasses, numFeatures]);
        const classPriors = new Float32Array(numClasses);

        this.params = {
            type: 'gaussian',
            classes,
            classMeans,
            classVariances,
            classPriors,
        };

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

            const rowOffset = clsIndex * numFeatures;

            // Calculate mean for each feature
            for (let j = 0; j < numFeatures; j++) {
                let sum = 0;
                for (const idx of classIndices) {
                    sum += XArray[idx][j];
                }
                classMeans.array[rowOffset + j] = sum / classCount;
            }

            // Calculate variance for each feature
            for (let j = 0; j < numFeatures; j++) {
                let sumSq = 0;
                const mean = classMeans.array[rowOffset + j];
                for (const idx of classIndices) {
                    const diff = XArray[idx][j] - mean;
                    sumSq += diff * diff;
                }

                // Add regularization to prevent zero variance and overfitting
                classVariances.array[rowOffset + j] = sumSq / classCount + this.varianceSmoothing;
            }

            this.params = {
                type: 'gaussian',
                classes,
                classMeans,
                classVariances,
                classPriors,
            };

            await this.eventEmitter?.emit('callback', {
                threadId: 0,
                iteration: clsIndex,
                params: this.params,
            });
        }

        return this.params;
    }

    /**
     * Predicts class labels for input features.
     *
     * @param X - Input features tensor of shape [n_samples, n_features]
     * @param params - Optional model parameters (uses trained params if not provided)
     * @returns Tensor of predicted class indices of shape [n_samples, 1]
     */
    predict(X: Tensor2D, params?: GaussianNaiveBayesParams): Tensor2D {
        const modelParams = params ?? this.params;
        assertModelTrained(modelParams);

        const XArray = X.arraySync();
        const numSamples = XArray.length;
        const predictions: number[][] = [];

        for (let i = 0; i < numSamples; i++) {
            const sample = XArray[i];
            const logProbs = this.calculateLogProbabilities(sample, modelParams);
            const maxIdx = logProbs.indexOf(Math.max(...logProbs));
            predictions.push([modelParams.classes[maxIdx]]);
        }

        return tensor2d(predictions);
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
        params?: GaussianNaiveBayesParams,
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
            const logProbs = this.calculateLogProbabilities(sample, modelParams!);

            // Convert log probabilities to probabilities
            const maxLogProb = Math.max(...logProbs);
            const expProbs = logProbs.map((lp) => Math.exp(lp - maxLogProb));
            const sumExpProbs = expProbs.reduce((a, b) => a + b, 0);
            const probs = expProbs.map((p) => p / sumExpProbs);

            const maxIdx = logProbs.indexOf(maxLogProb);
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
     * Calculates log probabilities for each class for a given sample.
     *
     * @param sample - Feature vector for a single sample
     * @param params - Model parameters
     * @returns Array of log probabilities for each class
     */
    private calculateLogProbabilities(
        sample: number[],
        params: GaussianNaiveBayesParams,
    ): Float32Array {
        const { classes, classPriors, classMeans, classVariances } = params;
        const numClasses = classes.length;

        const logProbs = new Float32Array(numClasses);

        for (let c = 0; c < numClasses; c++) {
            let logProb = Math.log(classPriors[c]);

            const rowOffset = c * sample.length;

            // Calculate log probability for each feature using Gaussian PDF
            for (let j = 0; j < sample.length; j++) {
                const mean = classMeans.array[rowOffset + j];
                const variance = classVariances.array[rowOffset + j];
                const x = sample[j];

                // Log of Gaussian PDF: log(1/sqrt(2πσ²)) - (x-μ)²/(2σ²)
                const logPdf =
                    -0.5 * Math.log(2 * Math.PI * variance) - (x - mean) ** 2 / (2 * variance);

                logProb += logPdf;
            }
            logProbs[c] = logProb;
        }

        return logProbs;
    }
}
