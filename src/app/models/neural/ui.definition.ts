import type { ModelDefinition } from '@/app/shared/registry/types';
import type { TaskType } from '@/app/shared/types';
import { NeuralSettings } from './ui/NeuralSettings';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { NeuralMainMetrics } from './ui/NeuralMainMetrics';
import { LossHistoryPlot } from '@/app/shared/visualization';
import { NeuralModelDataPlots } from './ui/NeuralModelDataPlots';

export const neuralModelDefinition: ModelDefinition<'neural'> = {
    key: 'neural',
    label: 'Neural Networks',
    taskTypes: ['regression', 'classification'],
    defaultSettings: (taskType?: TaskType) => ({
        type: 'neural',
        lossFunction: { type: taskType === 'regression' ? 'mse' : 'binaryCrossentropy' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'xavierNormal' },
        layers: [{ units: 1, activation: 'linear' }],
    }),
    settingsComponent: NeuralSettings,

    visualization: {
        metricsGridComponent: NeuralMainMetrics,
        modelDataPlotComponent: NeuralModelDataPlots,
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
