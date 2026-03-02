import type { ModelDefinition } from '@/app/shared/registry/types';
import {
    LossHistory,
    ConfusionMatrix,
    RocCurve,
    ResidualsPlot,
    RegressionMetrics,
} from '@/app/shared/visualization';
import { NeuralSettings } from './ui/NeuralSettings';
import { NeuralMainMetrics } from './ui/NeuralMainMetrics';
import { NeuralModelDataPlots } from './ui/NeuralModelDataPlots';
import {
    DEFAULT_CLASSIFICATION_REPORT,
    DEFAULT_CLASSIFICATION_SETTINGS,
    DEFAULT_REGRESSION_REPORT,
    DEFAULT_REGRESSION_SETTINGS,
} from './defaults';

export const neuralModelDefinition: ModelDefinition<'neural'> = {
    key: 'neural',
    label: 'Neural Networks',
    taskTypes: ['regression', 'classification'],
    defaultSettings: (taskType) =>
        taskType === 'regression' ? DEFAULT_REGRESSION_SETTINGS : DEFAULT_CLASSIFICATION_SETTINGS,
    settingsComponent: NeuralSettings,

    defaultReport: (taskType) => {
        switch (taskType) {
            case 'classification':
                return DEFAULT_CLASSIFICATION_REPORT;
            case 'regression':
                return DEFAULT_REGRESSION_REPORT;
            default:
                throw new Error(`Unsupported task type: ${taskType}`);
        }
    },
    visualization: {
        metricsGridComponent: NeuralMainMetrics,
        modelDataPlotComponent: NeuralModelDataPlots,
        plots: (taskType) =>
            taskType === 'regression'
                ? [
                      { title: 'Loss History', component: LossHistory },
                      { title: 'Metrics', component: RegressionMetrics },
                      { title: 'Residuals', component: ResidualsPlot },
                  ]
                : [
                      { title: 'Loss History', component: LossHistory },
                      { title: 'Confusion Matrix', component: ConfusionMatrix },
                      { title: 'ROC Curve', component: RocCurve },
                  ],
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
