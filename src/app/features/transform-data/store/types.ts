import type { Transformation } from '@/app/shared/types';

export type NormalizationMethod = 'none' | 'zscore' | 'linear' | 'log';

export type TransformationSettings = {
    normalization: NormalizationMethod;
    transformations: Transformation[];
};
