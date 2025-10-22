import type { ModelDefinition } from '@/app/shared/registry/types';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { LinearSettings } from './ui/LinearSettings';
import { LinearMainMetrics } from './ui/LinearMainMetrics';
import { LossHistoryPlot, LinearPlots } from '@/app/shared/visualization';

export const linearModelDefinition: ModelDefinition<'linear'> = {
    key: 'linear',
    label: 'Linear Regression',
    taskTypes: ['regression'],
    defaultSettings: () => ({
        type: 'linear',
        lossFunction: { type: 'mse' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'zeros' },
    }),
    settingsComponent: LinearSettings,

    visualization: {
        metricsGridComponent: LinearMainMetrics,
        modelDataPlotComponent: LinearPlots,
        plots: [{ title: 'Loss History', component: LossHistoryPlot }],
    },

    progress: {
        getProgressInfo: (report, settings) => {
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
