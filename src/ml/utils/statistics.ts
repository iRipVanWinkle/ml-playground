import { Matrix, type MatrixLike } from './matrix';

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
    indices?: ArrayLike<number>,
): Float32Array {
    const means = new Float32Array(numFeatures);
    const numSamples = indices?.length ?? X.length;

    if (numSamples === 0) return means;

    for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
        let sum = 0;

        for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            const sample = X[indices?.[sampleIndex] ?? sampleIndex];
            sum += sample[featureIndex];
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
    X: ArrayLike<ArrayLike<number>>,
    means: ArrayLike<number>,
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
    X: ArrayLike<ArrayLike<number>>,
    means: ArrayLike<number>,
    numFeatures: number,
    indices?: ArrayLike<number>,
): Matrix {
    const covariance = new Float32Array(numFeatures * numFeatures);
    const numSamples = indices?.length ?? X.length;

    if (numSamples === 0) {
        return new Matrix({ array: covariance, shape: [numFeatures, numFeatures] });
    }

    const diffs = new Float32Array(numFeatures);

    for (let i = 0; i < numSamples; i++) {
        const row = X[indices?.[i] ?? i];

        for (let j = 0; j < numFeatures; j++) {
            diffs[j] = row[j] - means[j];
        }

        for (let rowIdx = 0; rowIdx < numFeatures; rowIdx++) {
            const rowOffset = rowIdx * numFeatures;
            const diffRow = diffs[rowIdx];

            for (let colIdx = rowIdx; colIdx < numFeatures; colIdx++) {
                covariance[rowOffset + colIdx] += diffRow * diffs[colIdx];
            }
        }
    }

    for (let rowIdx = 0; rowIdx < numFeatures; rowIdx++) {
        const rowOffset = rowIdx * numFeatures;
        for (let colIdx = 0; colIdx < rowIdx; colIdx++) {
            covariance[rowOffset + colIdx] = covariance[colIdx * numFeatures + rowIdx];
        }
    }

    for (let i = 0; i < covariance.length; i++) {
        covariance[i] /= numSamples;
    }

    return new Matrix({ array: covariance, shape: [numFeatures, numFeatures] });
}

/**
 * Calculates the log-PDF of a diagonal (independent features) Gaussian distribution.
 * Equivalent to the sum of univariate Gaussian log-PDFs for each feature.
 *
 * @param sample - Feature vector for a single sample
 * @param means - Mean of each feature
 * @param variances - Variance of each feature
 * @returns The total log-probability density
 */
export function calculateDiagonalGaussianLogPdf(
    sample: ArrayLike<number>,
    means: ArrayLike<number>,
    variances: ArrayLike<number>,
): number {
    let logProb = 0;
    for (let j = 0; j < sample.length; j++) {
        const mean = means[j];
        const variance = variances[j];
        const x = sample[j];

        // Log of Gaussian PDF: log(1/√(2πσ²)) - (x-μ)²/(2σ²)
        const logPdf = -0.5 * Math.log(2 * Math.PI * variance) - (x - mean) ** 2 / (2 * variance);
        logProb += logPdf;
    }

    return logProb;
}

/**
 * Calculates the log-PDF of a multivariate Gaussian distribution with full covariance.
 *
 * @param sample - Feature vector for a single sample
 * @param means - Mean of each feature
 * @param covarianceInverse - Inverse of the covariance matrix (MatrixLike)
 * @param covarianceDeterminant - Determinant of the covariance matrix
 * @returns The log-probability density
 */
export function calculateFullGaussianLogPdf(
    sample: ArrayLike<number>,
    means: ArrayLike<number>,
    covarianceInverse: MatrixLike,
    covarianceDeterminant: number,
): number {
    const numFeatures = sample.length;
    const diff = new Float32Array(numFeatures);
    for (let i = 0; i < numFeatures; i++) {
        diff[i] = sample[i] - means[i];
    }

    // Compute Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ)
    let mahalanobis = 0;
    for (let i = 0; i < numFeatures; i++) {
        let sum = 0;
        for (let j = 0; j < numFeatures; j++) {
            sum += covarianceInverse.array[i * numFeatures + j] * diff[j];
        }
        mahalanobis += diff[i] * sum;
    }

    // log N(x|μ,Σ) = -0.5 * [d·log(2π) + log(|Σ|) + mahalanobis]
    const logPdf =
        -0.5 *
        (numFeatures * Math.log(2 * Math.PI) + Math.log(covarianceDeterminant) + mahalanobis);

    return logPdf;
}
