import { Matrix } from '../matrix';

/**
 * Calculates the mean for each feature in a dataset.
 *
 * @param X - Features array of shape [n_samples, n_features]
 * @param numFeatures - Number of features (columns) in X
 * @param indices - Optional array of sample indices to include in the calculation.
 *                  If provided, only these samples will be used. This is useful
 *                  for calculating class-specific means without copying data.
 * @returns Float32Array containing the mean of each feature
 */
export function calculateMean(
    X: number[][],
    numFeatures: number,
    indices?: number[],
): Float32Array {
    const means = new Float32Array(numFeatures);
    const numSamples = indices?.length ?? X.length;

    if (numSamples === 0) return means;

    for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
        let sum = 0;

        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sample = indices?.[sampleIndex] ?? sampleIndex;
            sum += X[sample][featureIndex];
        }

        means[featureIndex] = sum / numSamples;
    }

    return means;
}

/**
 * Calculates the variance for each feature in a dataset.
 *
 * @param X - Features array of shape [n_samples, n_features]
 * @param means - Pre-calculated means of shape [n_features]
 * @param numFeatures - Number of features (columns) in X
 * @param indices - Optional array of sample indices to include
 * @returns Float32Array containing the variance of each feature
 */
export function calculateVariance(
    X: Float32Array[] | number[][],
    means: Float32Array,
    numFeatures: number,
    indices?: number[],
): Float32Array {
    const variances = new Float32Array(numFeatures);
    const numSamples = indices?.length ?? X.length;

    if (numSamples === 0) return variances;

    for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sample = X[indices?.[sampleIndex] ?? sampleIndex];
            const diff = sample[featureIndex] - means[featureIndex];
            variances[featureIndex] += diff * diff;
        }

        variances[featureIndex] /= numSamples;
    }

    return variances;
}

/**
 * Calculates the covariance matrix for a dataset.
 *
 * @param X - Features array of shape [n_samples, n_features]
 * @param means - Pre-calculated means of shape [n_features]
 * @param numFeatures - Number of features (columns) in X
 * @param indices - Optional array of sample indices to include
 * @returns A Matrix representing the [numFeatures x numFeatures] covariance matrix
 */
export function calculateCovarianceMatrix(
    X: Float32Array[] | number[][],
    means: Float32Array,
    numFeatures: number,
    indices?: number[],
): Matrix {
    const covariance = new Float32Array(numFeatures * numFeatures);
    const numSamples = indices?.length ?? X.length;

    if (numSamples === 0) {
        return new Matrix({ array: covariance, shape: [numFeatures, numFeatures] });
    }

    for (let i = 0; i < numSamples; i++) {
        const row = X[indices?.[i] ?? i];

        for (let rowIdx = 0; rowIdx < numFeatures; rowIdx++) {
            const rowOffset = rowIdx * numFeatures;
            const diffRow = row[rowIdx] - means[rowIdx];

            for (let colIdx = 0; colIdx < numFeatures; colIdx++) {
                const diffCol = row[colIdx] - means[colIdx];
                covariance[rowOffset + colIdx] += diffRow * diffCol;
            }
        }
    }

    for (let i = 0; i < covariance.length; i++) {
        covariance[i] /= numSamples;
    }

    return new Matrix({ array: covariance, shape: [numFeatures, numFeatures] });
}
