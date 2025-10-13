import { initState, useTransformationStore } from './store';
import type { TransformationSettings } from './types';

export function updateTransformations(transformations: TransformationSettings['transformations']) {
    useTransformationStore.setState({ transformations });
}

export function resetTransformations() {
    useTransformationStore.setState({ transformations: initState.transformations });
}

export function updateNormalization(normalization: TransformationSettings['normalization']) {
    useTransformationStore.setState({ normalization });
}

export function resetNormalization() {
    useTransformationStore.setState({ normalization: initState.normalization });
}
