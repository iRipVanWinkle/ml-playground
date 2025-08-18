import { concat, tidy, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Extends data set with sinusoid features.
 *
 * Returns a new tf.Tensor2D with more features, comprising of sin(x).
 *
 * @param dataset - tf.Tensor2D.
 * @param sinusoidDegree - Multiplier for sinusoid parameter multiplications.
 * @returns The new tf.Tensor2D with the sinusoid features added.
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
