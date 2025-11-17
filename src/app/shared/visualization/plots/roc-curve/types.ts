import type { MulticlassRocCurve, RocCurve } from '@/ml/metrics';

export type BinaryRocCurveData = RocCurve & {
    type: 'binary';
    auc: number;
};

export type MulticlassRocCurveData = MulticlassRocCurve & {
    type: 'multiclass';
    aucs: number[];
    macroAuc: number;
    weightedAuc: number;
};

export type RocCurveData = BinaryRocCurveData | MulticlassRocCurveData;
