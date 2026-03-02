import type { ModelDefinition } from '@/app/shared/registry/types';
import { KMeansPlots } from '@/app/shared/visualization/plots/k-means/KMeansPlots';
import { InertiaHistory, KMeansMetrics, KMeansParameters } from '@/app/shared/visualization';
import { KMeansSettings } from './ui/KMeansSettings';
import { KMeansMainMetrics } from './ui/KMeansMainMetrics';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';

export const kMeansModelDefinition: ModelDefinition<'k-means'> = {
    key: 'k-means',
    label: 'K-Means',
    taskTypes: ['clustering'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: KMeansSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: KMeansMainMetrics,
        modelDataPlotComponent: KMeansPlots,
        plots: [
            { title: 'Inertia History', component: InertiaHistory },
            { title: 'Metrics', component: KMeansMetrics },
        ],
        parametersComponent: KMeansParameters,
    },

    progress: {
        getProgressInfo: ({ report, settings }) => {
            const currentIteration = report?.iteration ?? 0;
            const maxIteration = settings.maxIterations;
            return {
                type: 'determinate',
                label: `${currentIteration}/${maxIteration}`,
                current: currentIteration,
                max: maxIteration,
            };
        },
    },
};
