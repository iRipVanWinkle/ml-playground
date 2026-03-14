import type { ModelDefinition } from '@/app/shared/registry/types';
import { DBSCANSettings } from './ui/DBSCANSettings';
import { DBSCANMainMetrics } from './ui/DBSCANMainMetrics';
import { DBSCANPlots } from './ui/DBSCANPlots';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';

export const dbscanModelDefinition: ModelDefinition<'dbscan'> = {
    key: 'dbscan',
    label: 'DBSCAN',
    taskTypes: ['clustering', 'anomaly'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: DBSCANSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: DBSCANMainMetrics,
        modelDataPlotComponent: DBSCANPlots,
    },

    progress: {
        getProgressInfo: ({ report }) => ({
            type: 'indeterminate',
            label: report.numClusters > 0 ? `${report.numClusters} clusters` : 'Waiting...',
        }),
    },
};
