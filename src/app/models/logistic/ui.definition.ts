import type { ModelDefinition } from '@/app/shared/registry/types';
import {
    LogisticPlots,
    LossHistory,
    ConfusionMatrix,
    RocCurve,
    RegressionParameters,
    AccuracyMetricsDisplay,
} from '@/app/shared/visualization';
import { LogisticSettings } from './ui/LogisticSettings';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';

function arrayAvg(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((acc, val) => acc + val, 0) / arr.length;
}

export const logisticModelDefinition: ModelDefinition<'logistic'> = {
    key: 'logistic',
    label: 'Logistic Regression',
    taskTypes: ['classification'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: LogisticSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: AccuracyMetricsDisplay,
        modelDataPlotComponent: LogisticPlots,
        plots: [
            { title: 'Loss History', component: LossHistory },
            { title: 'Confusion Matrix', component: ConfusionMatrix },
            { title: 'ROC Curve', component: RocCurve },
        ],
        parametersComponent: RegressionParameters,
    },

    progress: {
        getProgressInfo: ({ report, settings }) => {
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
