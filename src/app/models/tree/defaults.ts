import { createEmptyMatrix } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_REGRESSION_SETTINGS: SettingsOf<'tree'> = {
    type: 'tree',
    modelVariant: 'decision',
    criterion: { type: 'mse' },
    maxDepth: 5,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    estimators: 10,
    maxFeatures: 1,
    numRandomThresholds: 1,
};

export const DEFAULT_CLASSIFICATION_SETTINGS: SettingsOf<'tree'> = {
    type: 'tree',
    modelVariant: 'decision',
    criterion: { type: 'gini' },
    maxDepth: 5,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    estimators: 10,
    maxFeatures: 1,
    numRandomThresholds: 1,
};

export const DEFAULT_REGRESSION_REPORT: TrainingReportOf<'tree'> = {
    type: 'tree',
    taskType: 'regression',
    trainPredictedLabels: createEmptyMatrix(),
    trainMetrics: null,
    trainResiduals: createEmptyMatrix(),
    params: [],
};

export const DEFAULT_CLASSIFICATION_REPORT: TrainingReportOf<'tree'> = {
    type: 'tree',
    taskType: 'classification',
    trainPredictedLabels: createEmptyMatrix(),
    params: [],
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
