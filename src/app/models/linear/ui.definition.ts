import type { ModelDefinition } from '@/app/shared/registry/types';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { LinearSettings } from './ui/LinearSettings';
import { LinearMainMetrics } from './ui/LinearMainMetrics';
import { EMPTY_MATRIX_LIKE } from '@/ml/matrix';
import {
    LinearPlots,
    LossHistory,
    RegressionMetrics,
    ResidualsPlot,
} from '@/app/shared/visualization';

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

    defaultReport: () => ({
        type: 'linear',
        taskType: 'regression',
        trainLossHistory: [],
        iteration: 0,
        trainLoss: 0,
        testLoss: 0,
        trainPredictedLabels: EMPTY_MATRIX_LIKE,
        theta: EMPTY_MATRIX_LIKE,
        trainMetrics: null,
        trainResiduals: EMPTY_MATRIX_LIKE,
    }),
    visualization: {
        metricsGridComponent: LinearMainMetrics,
        modelDataPlotComponent: LinearPlots,
        plots: [
            { title: 'Loss History', component: LossHistory },
            { title: 'Metrics', component: RegressionMetrics },
            { title: 'Residuals', component: ResidualsPlot },
        ],
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
