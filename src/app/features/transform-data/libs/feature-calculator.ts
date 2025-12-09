import type { TransformationType } from '@/app/shared/types';
import { calculateOutputFeatures } from '@/app/shared/helpers';

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
