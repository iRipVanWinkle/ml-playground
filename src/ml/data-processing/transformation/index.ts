import type { Tensor2D } from '@tensorflow/tfjs';
import { generateSinusoidalFeatures } from './generateSinusoidFeatures';
import { generateFullPolynomialFeatures } from './generateFullPolynomialFeatures';
import { generateCosinusoidalFeatures } from './generateCosinusoidalFeatures';
import { generateFourierFeatures } from './generateFourierFeatures';

export type TransformationFn = (data: Tensor2D) => Tensor2D | null;

export function cosinusoidGenerator(degree: number): TransformationFn {
    return (data: Tensor2D): Tensor2D => generateCosinusoidalFeatures(data, degree);
}

export function sinusoidGenerator(degree: number): TransformationFn {
    return (data: Tensor2D): Tensor2D => generateSinusoidalFeatures(data, degree);
}

export function fourierGenerator(degree: number): TransformationFn {
    return (data: Tensor2D): Tensor2D => generateFourierFeatures(data, degree);
}

export function fullPolynomialGenerator(degree: number): TransformationFn {
    return (data: Tensor2D): Tensor2D | null => generateFullPolynomialFeatures(data, degree);
}
