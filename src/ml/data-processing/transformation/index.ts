import type { Tensor2D } from '@tensorflow/tfjs';
import type { NormalizatorFn } from '../normalization';
import { generateSinusoidalFeatures } from './generateSinusoidFeatures';
import { generateFullPolynomialFeatures } from './generateFullPolynomialFeatures';

export * from './calculateOutputFeatures';

export type TransformationFn = (data: Tensor2D) => Tensor2D | null;

export function sinusoidGenerator(degree: number): TransformationFn {
    return (data: Tensor2D): Tensor2D => generateSinusoidalFeatures(data, degree);
}

export function fullPolynomialGenerator(
    degree: number,
    normalizeFunction: NormalizatorFn,
): TransformationFn {
    return (data: Tensor2D): Tensor2D | null =>
        generateFullPolynomialFeatures(data, degree, normalizeFunction);
}
