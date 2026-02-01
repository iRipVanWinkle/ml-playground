import type { ModelDefinition } from '@/app/shared/registry/types';
import type { TaskType } from '@/app/shared/types';
import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { NeuralSettings } from './ui/NeuralSettings';
import { NeuralMainMetrics } from './ui/NeuralMainMetrics';
import { NeuralModelDataPlots } from './ui/NeuralModelDataPlots';
import {
    LossHistory,
    ConfusionMatrix,
    RocCurve,
    ResidualsPlot,
    RegressionMetrics,
} from '@/app/shared/visualization';

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
                    trainPredictedLabels: EMPTY_MATRIX_LIKE,
                    theta: EMPTY_MATRIX_LIKE,
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
                    trainRocCurve: {
                        type: 'binary',
                        auc: 0,
                        fpr: new Float32Array([]),
                        tpr: new Float32Array([]),
                        thresholds: new Float32Array([]),
                        youdenOptimalIndex: null,
                        closestToCornerIndex: null,
                    },
                };
            case 'regression':
                return {
                    type: 'neural',
                    taskType: 'regression',
                    trainLossHistory: [],
                    iteration: 0,
                    optimizerLoss: 0,
                    trainLoss: 0,
                    trainPredictedLabels: EMPTY_MATRIX_LIKE,
                    theta: EMPTY_MATRIX_LIKE,
                    trainMetrics: null,
                    trainResiduals: EMPTY_MATRIX_LIKE,
                };
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
