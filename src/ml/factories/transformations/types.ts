export type TransformationFunction = 'sinusoid' | 'cosinusoid' | 'fourier' | 'polynomial';

export type TransformationConfig = Array<{
    type: TransformationFunction;
    degree: number;
}>;
