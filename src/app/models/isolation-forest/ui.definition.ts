import type { ModelDefinition } from '@/app/shared/registry/types';
import { IsolationForestSettings } from './ui/IsolationForestSettings';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';
import {
    AnomaliesMetricsDisplay,
    AnomalyPlots,
    DecisionTreeParameters,
} from '@/app/shared/visualization';

export const isolationForestModelDefinition: ModelDefinition<'isolation-forest'> = {
    key: 'isolation-forest',
    label: 'Isolation Forest',
    taskTypes: ['anomaly'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: IsolationForestSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: AnomaliesMetricsDisplay,
        modelDataPlotComponent: AnomalyPlots,
        parametersComponent: DecisionTreeParameters,
    },

    progress: {
        getProgressInfo: ({ report, settings }) => ({
            type: 'determinate',
            label: `${report.type === 'isolation-forest' ? report.params.length : 0}/${settings.estimators}`,
            current: report.type === 'isolation-forest' ? report.params.length : 0,
            max: settings.estimators,
        }),
    },
};
