import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';
import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';

export const DEFAULT_SETTINGS: SettingsOf<'knn'> = {
    type: 'knn',
    k: 5,
    weights: 'uniform',
    distance: { type: 'euclidean' },
};

export const DEFAULT_REGRESSION_SETTINGS: SettingsOf<'knn'> = {
    type: 'knn',
    k: 5,
    weights: 'uniform',
    distance: { type: 'euclidean' },
};

export const DEFAULT_CLASSIFICATION_SETTINGS: SettingsOf<'knn'> = {
    type: 'knn',
    k: 5,
    weights: 'uniform',
    distance: { type: 'euclidean' },
};

export const DEFAULT_CLASSIFICATION_REPORT: TrainingReportOf<'knn'> = {
    type: 'knn',
    taskType: 'classification',
    trainAccuracy: 0,
    testAccuracy: 0,
    trainPredictedLabels: EMPTY_MATRIX_LIKE,
    trainConfusionMatrix: {
        matrix: [],
        metrics: {
            type: 'binary',
            accuracy: 0,
            mcc: 0,
            cohensKappa: 0,
            precision: 0,
            recall: 0,
            f1: 0,
        },
    },
    trainRocCurve: {
        type: 'binary',
        auc: 0,
        fpr: new Float32Array([]),
        tpr: new Float32Array([]),
        thresholds: new Float32Array([]),
        youdenOptimalIndex: null,
        closestToCornerIndex: null,
    },
};

export const DEFAULT_REGRESSION_REPORT: TrainingReportOf<'knn'> = {
    type: 'knn',
    taskType: 'regression',
    trainLoss: 0,
    trainPredictedLabels: EMPTY_MATRIX_LIKE,
    trainMetrics: null,
    trainResiduals: EMPTY_MATRIX_LIKE,
};
