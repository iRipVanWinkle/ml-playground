import type { Tensor2D } from '@tensorflow/tfjs';
import type { DiagonalGaussianDistributionParams } from '../../types';
import { calculateMean, calculateVariance, calculateDiagonalGaussianLogPdf } from '../../utils';
import {
    BaseGaussianDistribution,
    type BaseGaussianDistributionOptions,
} from './BaseGaussianDistribution';

/**
 * Diagonal Gaussian Distribution model for anomaly detection.
 *
 * Assumes features are independent (diagonal covariance matrix).
 * Identifies anomalies based on probability density threshold.
 */
export class DiagonalGaussianDistribution extends BaseGaussianDistribution<DiagonalGaussianDistributionParams> {
    constructor(options: BaseGaussianDistributionOptions) {
        super(options);
    }

    /**
     * Trains the model by estimating per-feature mean and variance.
     */
    async train(X: Tensor2D): Promise<DiagonalGaussianDistributionParams> {
        const [, numFeatures] = X.shape;
        const XArray = await X.array();

        const featureMeans = calculateMean(XArray, numFeatures);

        const featureVariances = calculateVariance(XArray, featureMeans, numFeatures);
        for (let j = 0; j < numFeatures; j++) {
            featureVariances[j] += this.varianceSmoothing;
        }

        this.params = {
            type: 'gaussian-distribution',
            covarianceType: 'diagonal',
            featureMeans,
            featureVariances,
            threshold: this.threshold,
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
        params: DiagonalGaussianDistributionParams,
    ): number {
        return Math.exp(
            calculateDiagonalGaussianLogPdf(sample, params.featureMeans, params.featureVariances),
        );
    }
}
