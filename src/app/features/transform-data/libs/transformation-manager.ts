import type {
    AnyTransformation,
    DraftTransformation,
    TransformationType,
} from '@/app/shared/types';

export function createEmptyTransformation(): DraftTransformation {
    return { type: '', degree: 1 };
}

export function updateTransformationType(
    transformations: AnyTransformation[],
    index: number,
    type: TransformationType,
): AnyTransformation[] {
    const updated = [...transformations];
    updated[index] = { ...updated[index], type };
    return updated;
}

export function updateTransformationDegree(
    transformations: AnyTransformation[],
    index: number,
    degree: number,
): AnyTransformation[] {
    const updated = [...transformations];
    updated[index] = { ...updated[index], degree };
    return updated;
}

export function removeTransformation(
    transformations: AnyTransformation[],
    index: number,
): AnyTransformation[] {
    return transformations.filter((_, i) => i !== index);
}
