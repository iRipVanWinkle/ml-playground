import type { ModelDefinition } from '@/app/shared/registry/types';
import { ConfusionMatrix, RegressionMetrics, ResidualsPlot } from '@/app/shared/visualization';
import { RocCurve } from '@/app/shared/visualization';
import { KNNSettings } from './ui/KNNSettings';
import { KNNMainMetrics } from './ui/KNNMainMetrics';
import { KNNModelDataPlots } from './ui/KNNModelDataPlots';
import {
    DEFAULT_CLASSIFICATION_REPORT,
    DEFAULT_CLASSIFICATION_SETTINGS,
    DEFAULT_REGRESSION_REPORT,
    DEFAULT_REGRESSION_SETTINGS,
} from './defaults';

export const knnModelDefinition: ModelDefinition<'knn'> = {
    key: 'knn',
    label: 'K-Nearest Neighbors',
    taskTypes: ['classification', 'regression'],

    defaultSettings: (taskType) =>
        taskType === 'regression' ? DEFAULT_REGRESSION_SETTINGS : DEFAULT_CLASSIFICATION_SETTINGS,

    settingsComponent: KNNSettings,

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
        metricsGridComponent: KNNMainMetrics,
        modelDataPlotComponent: KNNModelDataPlots,
        plots: (taskType) =>
            taskType === 'regression'
                ? [
                      { title: 'Metrics', component: RegressionMetrics },
                      { title: 'Residuals', component: ResidualsPlot },
                  ]
                : [
                      { title: 'Confusion Matrix', component: ConfusionMatrix },
                      { title: 'ROC Curve', component: RocCurve },
                  ],
    },

    progress: {
        getProgressInfo: () => ({
            type: 'indeterminate',
            label: '',
        }),
    },
};
