import { createEmptyMatrix } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';
import { DEFAULT_OPTIMIZER } from '../defaults';

export const DEFAULT_SETTINGS: SettingsOf<'linear'> = {
    type: 'linear',
    lossFunction: { type: 'mse' },
    optimizer: DEFAULT_OPTIMIZER,
    regularization: { type: 'none' },
    thetaInitialization: { type: 'zeros' },
};

export const DEFAULT_REPORT: TrainingReportOf<'linear'> = {
    type: 'linear',
    taskType: 'regression',
    trainLossHistory: [],
    iteration: 0,
    optimizerLoss: 0,
    trainPredictedLabels: createEmptyMatrix(),
    theta: createEmptyMatrix(),
    trainMetrics: null,
    trainResiduals: createEmptyMatrix(),
};
