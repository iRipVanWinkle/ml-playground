import type { NormalizatorFn } from '@/ml/data-processing/normalization';
import {
    fullPolynomialGenerator,
    sinusoidGenerator,
    type TransformationFn,
} from '@/ml/data-processing/transformation';
import type { TransformationFunction } from '@/app/store';

export function getTransformations(
    transformationsConfig: {
        type: TransformationFunction;
        degree: number;
    }[],
    normalizeFunction?: NormalizatorFn,
): TransformationFn[] {
    const transformations = [];
    for (const transformation of transformationsConfig) {
        const { type, degree } = transformation;
        switch (type) {
            case 'sinusoid':
                transformations.push(sinusoidGenerator(degree));
                break;
            case 'polynomial':
                transformations.push(
                    fullPolynomialGenerator(degree, normalizeFunction ?? ((v) => v)),
                );
                break;
            default:
                console.warn(`Unknown transformation type: ${transformation.type}`);
        }
    }

    return transformations;
}
