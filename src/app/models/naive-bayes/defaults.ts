import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'naive-bayes'> = {
    type: 'naive-bayes',
    variant: 'gaussian',
};

export const DEFAULT_REPORT: TrainingReportOf<'naive-bayes'> = {
    type: 'naive-bayes',
    taskType: 'classification',
    testAccuracy: 0,
    trainAccuracy: 0,
    trainPredictedLabels: EMPTY_MATRIX_LIKE,
    iteration: 0,
    params: {
        type: 'gaussian',
        classes: [],
        classPriors: new Float32Array(),
        classMeans: EMPTY_MATRIX_LIKE,
        classVariances: EMPTY_MATRIX_LIKE,
    },
    trainConfusionMatrix: {
        matrix: [] as number[][],
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
