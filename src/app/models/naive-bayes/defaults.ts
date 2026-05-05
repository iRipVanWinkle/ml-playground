import { createEmptyMatrix } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'naive-bayes'> = {
    type: 'naive-bayes',
    variant: 'gaussian',
};

export const DEFAULT_REPORT: TrainingReportOf<'naive-bayes'> = {
    type: 'naive-bayes',
    taskType: 'classification',
    trainPredictedLabels: createEmptyMatrix(),
    iteration: 0,
    params: {
        type: 'gaussian',
        classes: [],
        classPriors: new Float32Array(),
        classMeans: createEmptyMatrix(),
        classVariances: createEmptyMatrix(),
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
