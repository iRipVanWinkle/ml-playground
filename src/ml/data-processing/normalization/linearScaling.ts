import { tidy, tensor1d, type Tensor, type Tensor2D, keep } from '@tensorflow/tfjs';
import type { Scaler, MinMaxScalerParams } from '../../types';
import { EPSILON } from '../../constants';

/**
 * Linear scaling using TensorFlow.js.
 * Scales the input tensor to the [0, 1] range using min-max normalization.
 */
export class MinMaxScaler implements Scaler<MinMaxScalerParams> {
    private min: Tensor | null = null;
    private max: Tensor | null = null;

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

            this.min = keep(tensor.min(0));
            this.max = keep(tensor.max(0));
        });
    }

    /**
     * Transforms the input tensor using min-max scaling.
     *
     * @param tensor - A Tensor2D representing the matrix to be scaled.
     * @returns A new Tensor2D with scaled values.
     */
    transform(tensor: Tensor2D): Tensor2D {
        return tidy(() => {
            if (!this.max || !this.min) {
                throw new Error('Scaler has not been fitted yet');
            }

            if (tensor.size === 0) {
                throw new Error('Input tensor is empty');
            }

            // Min-max scaling: (x - min) / (max - min)
            // Add EPSILON to avoid division by zero if all values in a feature are the same
            const range = this.max.sub(this.min);
            const safeRange = range.add(EPSILON);
            const scaled = tensor.sub(this.min).div(safeRange);
            return scaled as Tensor2D;
        });
    }

    /**
     * Extracts the parameters of the scaler.
     *
     * @returns A promise that resolves to the scaler parameters.
     */
    async extractParameters(): Promise<MinMaxScalerParams> {
        if (!this.min || !this.max) {
            throw new Error('Scaler has not been fitted yet');
        }

        const [min, max] = await Promise.all([
            this.min.data<'float32'>(),
            this.max.data<'float32'>(),
        ]);

        return {
            type: 'minmax',
            min,
            max,
        };
    }

    /**
     * Restores the parameters of the scaler.
     *
     * @param params - The scaler parameters to restore.
     */
    restoreParameters(params: MinMaxScalerParams): void {
        this.min = tensor1d(params.min);
        this.max = tensor1d(params.max);
    }

    /**
     * Disposes of any resources used by the scaler.
     */
    dispose(): void {
        this.min?.dispose();
        this.max?.dispose();
    }
}
