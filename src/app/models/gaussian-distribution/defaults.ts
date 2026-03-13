import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'gaussian-distribution'> = {
    type: 'gaussian-distribution',
    variant: 'diagonal',
    threshold: 0.01,
    varianceSmoothing: 1e-9,
};

export const DEFAULT_REPORT: TrainingReportOf<'gaussian-distribution'> = {
    type: 'gaussian-distribution',
    taskType: 'anomaly',
    trainAnomalyRate: 0,
    trainPredictions: EMPTY_MATRIX_LIKE,
    params: {
        type: 'gaussian-distribution',
        covarianceType: 'diagonal',
        featureMeans: new Float32Array(),
        featureVariances: new Float32Array(),
        threshold: 0.01,
    },
};
