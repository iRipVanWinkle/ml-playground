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

export type TypedArray = Float32Array | Uint32Array | Int32Array;

export type TypedMatrix = {
    array: TypedArray;
    shape: Uint8Array;
};
