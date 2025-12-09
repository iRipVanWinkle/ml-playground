import type { Transformation, TransformationType } from '@/app/shared/types';

export function createEmptyTransformation(): Transformation {
    return { type: '', degree: 1 } as unknown as Transformation;
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
