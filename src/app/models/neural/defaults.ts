import { createEmptyMatrix } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';
import { DEFAULT_OPTIMIZER } from '../defaults';

export const DEFAULT_REGRESSION_SETTINGS: SettingsOf<'neural'> = {
    type: 'neural',
    lossFunction: { type: 'mse' },
    optimizer: DEFAULT_OPTIMIZER,
    regularization: { type: 'none' },
    thetaInitialization: { type: 'xavierNormal' },
    layers: [{ units: 1, activation: 'linear' }],
};

export const DEFAULT_CLASSIFICATION_SETTINGS: SettingsOf<'neural'> = {
    type: 'neural',
    lossFunction: { type: 'binaryCrossentropy' },
    optimizer: DEFAULT_OPTIMIZER,
    regularization: { type: 'none' },
    thetaInitialization: { type: 'xavierNormal' },
    layers: [{ units: 1, activation: 'linear' }],
};

export const DEFAULT_REGRESSION_REPORT: TrainingReportOf<'neural'> = {
    type: 'neural',
    taskType: 'regression',
    trainLossHistory: [],
    iteration: 0,
    optimizerLoss: 0,
    trainPredictedLabels: createEmptyMatrix(),
    theta: createEmptyMatrix(),
    trainMetrics: null,
    trainResiduals: createEmptyMatrix(),
};

export const DEFAULT_CLASSIFICATION_REPORT: TrainingReportOf<'neural'> = {
    type: 'neural',
    taskType: 'classification',
    trainLossHistory: [],
    iteration: 0,
    trainPredictedLabels: createEmptyMatrix(),
    theta: createEmptyMatrix(),
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
