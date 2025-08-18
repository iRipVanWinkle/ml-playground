import type { NormalizatorFn } from '../normalization';
import { concat, ones, pow, tidy, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Extends data set with polynomial features of certain degree.
 *
 * Returns a new tf.Tensor2D with more features, comprising of
 * x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
 *
 * @param dataset - tf.Tensor2D to generate polynomials for.
 * @param polynomialDegree - The max power of new features.
 * @param normalizeData - Flag that indicates whether polynomials need to normalized or not.
 * @returns The new tf.Tensor2D with the polynomial features added.
 */
export function generateFullPolynomialFeatures(
    data: Tensor2D,
    degree: number,
    normalizeFunction: NormalizatorFn,
): Tensor2D | null {
    if (degree < 2) {
        return null; // No polynomial features generated for degree < 2;
    }

    const [numSamples, numFeatures] = data.shape;

    function generateExponentCombinations(n: number, d: number): number[][] {
        const results: number[][] = [];
        const combo: number[] = Array(n).fill(0);

        function recurse(pos: number, remaining: number) {
            if (pos === n - 1) {
                combo[pos] = remaining;
                results.push([...combo]);
                return;
            }
            for (let i = 0; i <= remaining; i++) {
                combo[pos] = i;
                recurse(pos + 1, remaining - i);
            }
        }

        recurse(0, d);
        return results;
    }

    return tidy(() => {
        const polynomialFeatures: Tensor2D[] = [];

        // Only generate additional features: degree ≥ 2
        for (let d = 2; d <= degree; d++) {
            const exponentsList = generateExponentCombinations(numFeatures, d);

            for (const exponents of exponentsList) {
                let term = ones([numSamples, 1]) as Tensor2D;

                for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                    const exp = exponents[featureIdx];
                    if (exp > 0) {
                        const featureCol = data.slice([0, featureIdx], [numSamples, 1]);
                        const powered = pow(featureCol, exp);
                        term = term.mul(powered) as Tensor2D;
                    }
                }

                polynomialFeatures.push(term);
            }
        }

        // Concatenate all generated features along the columns (axis=1)
        const concatenatedFeatures = concat(polynomialFeatures, 1) as Tensor2D;

        return normalizeFunction(concatenatedFeatures);
    });
}
