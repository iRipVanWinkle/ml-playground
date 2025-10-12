import { calculateOutputFeatures } from '@/ml/data-processing/transformation';
import type { TransformationType } from '../store/types';

export function calculateTransformationOutputFeatures(
    type: TransformationType,
    degree: number,
    numFeatures: number,
): number {
    return calculateOutputFeatures(type, degree, numFeatures);
}

export function isPolynomialWithDegreeOne(type: TransformationType, degree: number): boolean {
    return type === 'polynomial' && degree === 1;
}
