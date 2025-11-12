export interface MulticlassConfusionMatrixMetrics {
    type: 'multiclass';
    accuracy: number;
    mcc: number;
    cohensKappa: number;
    macroPrecision: number;
    macroRecall: number;
    macroF1: number;
    weightedPrecision: number;
    weightedRecall: number;
    weightedF1: number;
}

export interface BinaryConfusionMatrixMetrics {
    type: 'binary';
    accuracy: number;
    mcc: number;
    cohensKappa: number;
    precision: number;
    recall: number;
    f1: number;
}

export type ConfusionMatrixMetrics =
    | MulticlassConfusionMatrixMetrics
    | BinaryConfusionMatrixMetrics;

export type ConfusionMatrixData = {
    matrix: number[][];
    metrics: ConfusionMatrixMetrics;
    perClassMatrix?: number[][][];
    perClassMetrics?: BinaryConfusionMatrixMetrics[];
};
