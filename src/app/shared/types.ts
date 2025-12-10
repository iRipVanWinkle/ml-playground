export type TaskType = 'regression' | 'classification';

export type Dataset = {
    trainInputFeatures: number[][];
    trainTargetLabels: number[][];
    testInputFeatures: number[][];
    testTargetLabels: number[][];
    predictionInputFeatures?: number[][];
    xMin: number[];
    xMax: number[];
    headers: string[];
    categories?: string[];
    isImage?: boolean;
};

export type TransformationType = 'sinusoid' | 'cosinusoid' | 'fourier' | 'polynomial';

export type Transformation = {
    type: TransformationType;
    degree: number;
};
