import { concat, tidy, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Extends a dataset with sinusoidal (sin) features up to the given degree.
 *
 * @param data - Input feature tensor (samples × features).
 * @param degree - Number of frequency harmonics to generate (must be ≥ 1).
 * @returns A new Tensor2D with `degree * numFeatures` sin columns appended.
 */
export function generateSinusoidalFeatures(data: Tensor2D, degree: number): Tensor2D {
    if (degree < 1) {
        throw new Error('Degree must be at least 1');
    }

    return tidy(() => {
        const sinusoids: Tensor2D[] = [];

        for (let d = 1; d <= degree; d++) {
            sinusoids.push(data.mul(d).sin() as Tensor2D);
        }

        return concat(sinusoids, 1);
    });
}
