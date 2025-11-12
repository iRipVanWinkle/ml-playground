import type { ModelDefinition } from '@/app/shared/registry/types';
import type { TaskType } from '@/app/shared/types';
import { NeuralSettings } from './ui/NeuralSettings';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { NeuralMainMetrics } from './ui/NeuralMainMetrics';
import { ConfusionMatrix, LossHistoryPlot } from '@/app/shared/visualization';
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

    defaultReport: (taskType: TaskType) => {
        switch (taskType) {
            case 'classification':
                return {
                    type: 'neural',
                    taskType: 'classification',
                    trainLossHistory: [],
                    iteration: 0,
                    testAccuracy: 0,
                    trainAccuracy: 0,
                    trainPredictedLabels: [],
                    testPredictedLabels: [],
                    predictionPredictedLabels: [],
                    theta: [],
                    trainConfusionMatrix: {
                        matrix: [],
                        metrics: {
                            type: 'binary',
                            accuracy: 0,
                            mcc: 0,
                            cohensKappa: 0,
                            precision: 0,
                            recall: 0,
                            f1: 0,
                        },
                    },
                };
            case 'regression':
                return {
                    type: 'neural',
                    taskType: 'regression',
                    trainLossHistory: [],
                    iteration: 0,
                    trainLoss: 0,
                    testLoss: 0,
                    trainPredictedLabels: [],
                    testPredictedLabels: [],
                    predictionPredictedLabels: [],
                    theta: [],
                };
            default:
                throw new Error(`Unsupported task type: ${taskType}`);
        }
    },
    visualization: {
        metricsGridComponent: NeuralMainMetrics,
        modelDataPlotComponent: NeuralModelDataPlots,
        plots: [
            { title: 'Loss History', component: LossHistoryPlot },
            { title: 'Confusion Matrix', component: ConfusionMatrix },
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
