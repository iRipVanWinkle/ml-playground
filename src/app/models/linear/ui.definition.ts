import type { ModelDefinition } from '@/app/shared/registry/types';
import {
    LinearPlots,
    LossHistory,
    LossMetricsDisplay,
    RegressionMetrics,
    RegressionParameters,
    ResidualsPlot,
    RegressionPrediction,
} from '@/app/shared/visualization';
import { LinearSettings } from './ui/LinearSettings';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';

export const linearModelDefinition: ModelDefinition<'linear'> = {
    key: 'linear',
    label: 'Linear Regression',
    taskTypes: ['regression'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: LinearSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: LossMetricsDisplay,
        modelDataPlotComponent: LinearPlots,
        plots: [
            { title: 'Loss History', component: LossHistory },
            { title: 'Metrics', component: RegressionMetrics },
            { title: 'Residuals', component: ResidualsPlot },
        ],
        parametersComponent: RegressionParameters,
        predictionComponent: RegressionPrediction,
    },

    progress: {
        getProgressInfo: ({ report, settings }) => {
            const currentIteration = report?.iteration ?? 0;
            const maxIteration = settings.optimizer.maxIterations;
            return {
                type: 'determinate',
                label: `${currentIteration}/${maxIteration}`,
                current: currentIteration,
                max: maxIteration,
            };
        },
    },
};
