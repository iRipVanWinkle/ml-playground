import type { Transformation, TransformationType } from '../store/types';

export function createEmptyTransformation(): Transformation {
    return { type: 'sinusoid', degree: 1 };
}

export function updateTransformationType(
    transformations: Transformation[],
    index: number,
    type: TransformationType,
): Transformation[] {
    const updated = [...transformations];
    updated[index] = { ...updated[index], type };
    return updated;
}

export function updateTransformationDegree(
    transformations: Transformation[],
    index: number,
    degree: number,
): Transformation[] {
    const updated = [...transformations];
    updated[index] = { ...updated[index], degree };
    return updated;
}

export function removeTransformation(
    transformations: Transformation[],
    index: number,
): Transformation[] {
    return transformations.filter((_, i) => i !== index);
}
