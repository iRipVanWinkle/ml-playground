import type { ModelDefinition } from '@/app/shared/registry/types';
import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import { KMeansSettings } from './ui/KMeansSettings';
import { KMeansMainMetrics } from './ui/KMeansMainMetrics';
import { KMeansPlots } from '@/app/shared/visualization/plots/k-means/KMeansPlots';
import { InertiaHistory, KMeansMetrics, KMeansParameters } from '@/app/shared/visualization';

export const kMeansModelDefinition: ModelDefinition<'k-means'> = {
    key: 'k-means',
    label: 'K-Means',
    taskTypes: ['clustering'],
    defaultSettings: () => ({
        type: 'k-means',
        numClusters: 3,
        maxIterations: 100,
        centroidInitialization: { type: 'random' },
        distance: { type: 'euclidean' },
    }),
    settingsComponent: KMeansSettings,

    defaultReport: () => ({
        type: 'k-means',
        taskType: 'clustering',
        iteration: 0,
        trainAssignments: EMPTY_MATRIX_LIKE,
        centroids: EMPTY_MATRIX_LIKE,
        inertiaHistory: [],
    }),
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
        getProgressInfo: (report, settings) => {
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
