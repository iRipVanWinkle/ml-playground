import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'isolation-forest'> = {
    type: 'isolation-forest',
    estimators: 100,
    maxSamples: 256,
    contamination: 0.1,
    bootstrap: false,
};

export const DEFAULT_REPORT: TrainingReportOf<'isolation-forest'> = {
    type: 'isolation-forest',
    taskType: 'anomaly',
    scoreThreshold: 0.5,
    params: [],
};
