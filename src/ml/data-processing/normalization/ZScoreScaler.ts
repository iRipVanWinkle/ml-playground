import { moments, tidy, type Tensor, type Tensor2D, tensor1d, keep } from '@tensorflow/tfjs';
import type { Scaler, ZScoreScalerParams } from '../../types';
import { EPSILON } from '../../constants';

/**
 * Z-score normalization (standardization) implementation.
 *
 * This scaler transforms data to have a mean of 0 and a standard deviation of 1.
 * It computes the mean and standard deviation from the training data and applies
 * the transformation to both training and new data.
 */
export class ZScoreScaler implements Scaler<ZScoreScalerParams> {
    private mean: Tensor | null = null;
    private std: Tensor | null = null;

    /**
     * Fits the scaler to the input tensor.
     *
     * @param tensor - The input tensor.
     */
    fit(tensor: Tensor2D): void {
        tidy(() => {
            if (tensor.size === 0) {
                throw new Error('Input tensor is empty');
            }

            // Compute mean and std deviation along axis 0 (per feature/column)
            const { mean, variance } = moments(tensor, 0);
            const std = variance.sqrt();

            // Prevent division by zero
            const safeStd = std.add(EPSILON);

            this.mean = keep(mean);
            this.std = keep(safeStd);
        });
    }

    /**
     * Transforms the input tensor using z-score normalization.
     *
     * @param tensor - The input tensor to transform.
     * @returns The transformed tensor with mean 0 and standard deviation 1.
     */
    transform(tensor: Tensor2D): Tensor2D {
        return tidy(() => {
            if (!this.mean || !this.std) {
                throw new Error('Scaler has not been fitted yet');
            }

            if (tensor.size === 0) {
                throw new Error('Input tensor is empty');
            }

            // Center features
            const centered = tensor.sub(this.mean);

            // Normalize
            const scaled = centered.div(this.std);

            return scaled as Tensor2D;
        });
    }

    /**
     * Extracts the parameters of the scaler.
     *
     * @returns A promise that resolves to the scaler parameters.
     */
    async extractParameters(): Promise<ZScoreScalerParams> {
        if (!this.mean || !this.std) {
            throw new Error('Scaler has not been fitted yet');
        }

        const [mean, std] = await Promise.all([
            this.mean.data<'float32'>(),
            this.std.data<'float32'>(),
        ]);

        return {
            type: 'zscore',
            mean,
            std,
        };
    }

    /**
     * Restores the parameters of the scaler.
     *
     * @param params - The scaler parameters to restore.
     */
    restoreParameters(params: ZScoreScalerParams): void {
        this.std = tensor1d(params.std);
        this.mean = tensor1d(params.mean);
    }

    /**
     * Disposes of any resources used by the scaler.
     */
    dispose(): void {
        this.mean?.dispose();
        this.std?.dispose();
    }
}
