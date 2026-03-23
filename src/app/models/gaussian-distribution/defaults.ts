import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'gaussian-distribution'> = {
    type: 'gaussian-distribution',
    variant: 'diagonal',
    threshold: 0.001,
    varianceSmoothing: 0.000001,
};

export const DEFAULT_REPORT: TrainingReportOf<'gaussian-distribution'> = {
    type: 'gaussian-distribution',
    taskType: 'anomaly',
    params: {
        type: 'gaussian-distribution',
        covarianceType: 'diagonal',
        featureMeans: new Float32Array(),
        featureVariances: new Float32Array(),
    },
};
