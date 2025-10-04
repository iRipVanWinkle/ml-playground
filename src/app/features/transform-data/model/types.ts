export type NormalizationMethod = 'none' | 'zscore' | 'linear' | 'log';
export type TransformationType = 'sinusoid' | 'cosinusoid' | 'fourier' | 'polynomial';

export type Transformation = {
    type: TransformationType;
    degree: number;
};

export type TransformationSettings = {
    normalization: NormalizationMethod;
    transformations: Transformation[];
};
