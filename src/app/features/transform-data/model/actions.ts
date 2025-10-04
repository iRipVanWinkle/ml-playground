import { useTransformationSettings } from './store';
import type { TransformationSettings } from './types';

export function updateTransformations(transformations: TransformationSettings['transformations']) {
    useTransformationSettings.setState({ transformations });
}

export function updateNormalization(normalization: TransformationSettings['normalization']) {
    useTransformationSettings.setState({ normalization });
}
