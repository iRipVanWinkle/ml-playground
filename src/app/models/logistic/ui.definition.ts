import type { ModelDefinition } from '@/app/shared/registry/types';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { LogisticSettings } from './ui/LogisticSettings';
import { LogisticMainMetrics } from './ui/LogisticMainMetrics';
import { LossHistoryPlot, LogisticPlots } from '@/app/shared/visualization';

function arrayAvg(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((acc, val) => acc + val, 0) / arr.length;
}

export const logisticModelDefinition: ModelDefinition<'logistic'> = {
    key: 'logistic',
    label: 'Logistic Regression',
    taskTypes: ['classification'],
    defaultSettings: () => ({
        type: 'logistic',
        classificationType: 'binary',
        lossFunction: { type: 'binaryCrossentropy' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'zeros' },
    }),
    settingsComponent: LogisticSettings,

    defaultReport: () => ({
        type: 'logistic',
        taskType: 'classification',
        trainLossHistory: [],
        iterations: [],
        testAccuracy: 0,
        trainAccuracy: 0,
        trainPredictedLabels: [],
        testPredictedLabels: [],
        predictionPredictedLabels: [],
        theta: [],
    }),
    visualization: {
        metricsGridComponent: LogisticMainMetrics,
        modelDataPlotComponent: LogisticPlots,
        plots: [{ title: 'Loss History', component: LossHistoryPlot }],
    },

    progress: {
        getProgressInfo: (report, settings) => {
            const current = Math.round(arrayAvg(report?.iterations ?? []));
            const max = settings.optimizer.maxIterations;
            return {
                type: 'determinate',
                label: `${current}/${max}`,
                current,
                max,
            };
        },
    },
};
