import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type { GaussianNaiveBayesParams, PredictionMetadata } from '../../types';
import { BaseNaiveBayes, type BaseNaiveBayesOptions } from '../base/BaseNaiveBayes';
import { Matrix } from '../../utils/matrix';
import {
    calculateMean,
    calculateVariance,
    calculateDiagonalGaussianLogPdf,
    assertModelTrained,
} from '../../utils';

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

        // Pre-allocate buffer for class indices to optimize filtering loop
        const classIndicesBuffer = new Int32Array(numSamples);

        // Calculate statistics for each class
        for (let clsIndex = 0; clsIndex < classes.length; clsIndex++) {
            await this.trainingController?.handleControlFlow(true);

            if (this.trainingController?.isTrainingStopped) {
                break;
            }

            const cls = classes[clsIndex];

            // Filter samples belonging to this class
            let count = 0;
            for (let i = 0; i < numSamples; i++) {
                if (yArray[i] === cls) {
                    classIndicesBuffer[count++] = i;
                }
            }
            const classIndices = classIndicesBuffer.subarray(0, count);

            const classCount = classIndices.length;
            classPriors[clsIndex] = classCount / numSamples;

            const rowOffset = clsIndex * numFeatures;

            // Calculate mean for each feature
            const meanArr = calculateMean(XArray, numFeatures, classIndices);
            classMeans.row(clsIndex).set(meanArr);

            // Calculate variance for each feature
            const variancesArr = calculateVariance(XArray, meanArr, numFeatures, classIndices);
            for (let j = 0; j < numFeatures; j++) {
                // Add regularization to prevent zero variance and overfitting
                classVariances.array[rowOffset + j] = variancesArr[j] + this.varianceSmoothing;
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
        const resolvedParams = params ?? this.params;
        assertModelTrained(resolvedParams);

        const samplesArray = X.arraySync();
        const numSamples = samplesArray.length;
        const predictions = new Float32Array(numSamples);

        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sampleFeatures = samplesArray[sampleIndex];
            const classLogProbs = this.calculateLogProbabilities(sampleFeatures, resolvedParams);
            const predictedClassIndex = this.probabilityToClassIndex(classLogProbs);
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
    predictWithMetadata(X: Tensor2D, params?: GaussianNaiveBayesParams): PredictionMetadata {
        const resolvedParams = params ?? this.params;
        assertModelTrained(resolvedParams);

        const samplesArray = X.arraySync();
        const numSamples = samplesArray.length;
        const logProbabilitiesArray: number[][] = [];
        const predictedClassesArray: number[][] = [];

        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sampleFeatures = samplesArray[sampleIndex];
            const classLogProbs = this.calculateLogProbabilities(sampleFeatures, resolvedParams);
            const predictedClassIndex = this.probabilityToClassIndex(classLogProbs);

            predictedClassesArray.push([resolvedParams.classes[predictedClassIndex]]);
            logProbabilitiesArray.push([...classLogProbs]);
        }

        const probabilitiesTensor = tensor2d(logProbabilitiesArray);
        const predictionsTensor = tensor2d(predictedClassesArray);
        return {
            type: 'classification',
            predictions: predictionsTensor,
            probabilities: probabilitiesTensor,
            dispose() {
                predictionsTensor.dispose();
                probabilitiesTensor.dispose();
            },
        };
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
            const meansSlice = classMeans.array.subarray(
                c * sample.length,
                (c + 1) * sample.length,
            );
            const variancesSlice = classVariances.array.subarray(
                c * sample.length,
                (c + 1) * sample.length,
            );

            logProbs[c] =
                Math.log(classPriors[c]) +
                calculateDiagonalGaussianLogPdf(sample, meansSlice, variancesSlice);
        }

        return logProbs;
    }
}
