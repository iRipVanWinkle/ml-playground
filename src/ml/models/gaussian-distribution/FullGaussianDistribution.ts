import type { Tensor2D } from '@tensorflow/tfjs';
import type { FullGaussianDistributionParams } from '../../types';
import {
    calculateMean,
    calculateCovarianceMatrix,
    calculateInverseAndDeterminant,
    calculateFullGaussianLogPdf,
} from '../../utils';
import {
    BaseGaussianDistribution,
    type BaseGaussianDistributionOptions,
} from './BaseGaussianDistribution';

/**
 * Full-covariance Gaussian Distribution model for anomaly detection.
 *
 * Models the full covariance matrix to capture feature correlations.
 * Identifies anomalies based on probability density threshold.
 */
export class FullGaussianDistribution extends BaseGaussianDistribution<FullGaussianDistributionParams> {
    constructor(options: BaseGaussianDistributionOptions) {
        super(options);
    }

    /**
     * Trains the model by estimating mean, full covariance matrix, and its inverse.
     */
    async train(X: Tensor2D): Promise<FullGaussianDistributionParams> {
        const [, numFeatures] = X.shape;
        const XArray = await X.array();

        const featureMeans = calculateMean(XArray, numFeatures);

        const covarianceMatrix = calculateCovarianceMatrix(XArray, featureMeans, numFeatures);

        for (let i = 0; i < numFeatures; i++) {
            covarianceMatrix.array[i * numFeatures + i] += this.varianceSmoothing;
        }

        const { inverse, determinant } = calculateInverseAndDeterminant(covarianceMatrix);

        this.params = {
            type: 'gaussian-distribution',
            covarianceType: 'full',
            featureMeans,
            covarianceMatrix,
            covarianceInverse: inverse,
            covarianceDeterminant: determinant,
        };

        await this.eventEmitter?.emit('callback', {
            threadId: 0,
            iteration: 1,
            params: this.params,
        });

        return this.params;
    }

    protected calculateProbability(
        sample: number[],
        params: FullGaussianDistributionParams,
    ): number {
        return Math.exp(
            calculateFullGaussianLogPdf(
                sample,
                params.featureMeans,
                params.covarianceInverse,
                params.covarianceDeterminant,
            ),
        );
    }
}
