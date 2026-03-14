import type { ModelDefinition } from '@/app/shared/registry/types';
import { IsolationForestSettings } from './ui/IsolationForestSettings';
import { IsolationForestMainMetrics } from './ui/IsolationForestMainMetrics';
import { IsolationForestPlots } from './ui/IsolationForestPlots';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';
import { DecisionTreeParameters } from '@/app/shared/visualization';

export const isolationForestModelDefinition: ModelDefinition<'isolation-forest'> = {
    key: 'isolation-forest',
    label: 'Isolation Forest',
    taskTypes: ['anomaly'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: IsolationForestSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: IsolationForestMainMetrics,
        modelDataPlotComponent: IsolationForestPlots,
        parametersComponent: DecisionTreeParameters,
    },

    progress: {
        getProgressInfo: ({ report, settings }) => ({
            type: 'determinate',
            label: `${report.params.length}/${settings.estimators}`,
            current: report.params.length,
            max: settings.estimators,
        }),
    },
};
