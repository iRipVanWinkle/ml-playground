import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';
import { DEFAULT_OPTIMIZER } from '../defaults';

export const DEFAULT_SETTINGS: SettingsOf<'logistic'> = {
    type: 'logistic',
    classificationType: 'binary',
    lossFunction: { type: 'binaryCrossentropy' },
    optimizer: DEFAULT_OPTIMIZER,
    regularization: { type: 'none' },
    thetaInitialization: { type: 'zeros' },
};

export const DEFAULT_REPORT: TrainingReportOf<'logistic'> = {
    type: 'logistic',
    taskType: 'classification',
    trainLossHistory: [],
    iterations: [],
    testAccuracy: 0,
    trainAccuracy: 0,
    trainPredictedLabels: EMPTY_MATRIX_LIKE,
    theta: EMPTY_MATRIX_LIKE,
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
