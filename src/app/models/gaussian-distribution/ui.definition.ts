import type { ModelDefinition } from '@/app/shared/registry/types';
import { GaussianDistributionSettings } from './ui/GaussianDistributionSettings';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';
import {
    AnomaliesMetricsDisplay,
    AnomalyPlots,
    AnomalyPrediction,
} from '@/app/shared/visualization';

export const gaussianDistributionModelDefinition: ModelDefinition<'gaussian-distribution'> = {
    key: 'gaussian-distribution',
    label: 'Gaussian Distribution',
    taskTypes: ['anomaly'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: GaussianDistributionSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: AnomaliesMetricsDisplay,
        modelDataPlotComponent: AnomalyPlots,
        predictionComponent: AnomalyPrediction,
    },

    progress: {
        getProgressInfo: ({ report }) => ({
            type: 'determinate',
            label: (report.trainAnomalyRate ?? 0) > 0 ? 'Done' : '0/1',
            current: (report.trainAnomalyRate ?? 0) > 0 ? 1 : 0,
            max: 1,
        }),
    },
};
