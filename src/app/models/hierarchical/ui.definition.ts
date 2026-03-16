import type { ModelDefinition } from '@/app/shared/registry/types';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';
import { HierarchicalClusteringSettings } from './ui/HierarchicalClusteringSettings';
import { HierarchicalClusteringMainMetrics } from './ui/HierarchicalClusteringMainMetrics';
import { HierarchicalClusteringPlots } from './ui/HierarchicalClusteringPlots';

export const hierarchicalClusteringModelDefinition: ModelDefinition<'hierarchical'> = {
    key: 'hierarchical',
    label: 'Hierarchical Clustering',
    taskTypes: ['clustering'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: HierarchicalClusteringSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: HierarchicalClusteringMainMetrics,
        modelDataPlotComponent: HierarchicalClusteringPlots,
    },

    progress: {
        getProgressInfo: ({ report }) => ({
            type: 'indeterminate',
            label: report.numClusters > 0 ? `${report.numClusters} clusters` : 'Waiting...',
        }),
    },
};
