import type { ModelDefinition } from '@/app/shared/registry/types';
import { TreeSettings } from './ui/TreeSettings';
import { TreeMainMetrics } from './ui/TreeMainMetrics';
import { TreeModelDataPlots } from './ui/TreeModelDataPlots';
import {
    ConfusionMatrix,
    DecisionTreeParameters,
    RegressionMetrics,
    ResidualsPlot,
} from '@/app/shared/visualization';
import {
    DEFAULT_CLASSIFICATION_REPORT,
    DEFAULT_CLASSIFICATION_SETTINGS,
    DEFAULT_REGRESSION_REPORT,
    DEFAULT_REGRESSION_SETTINGS,
} from './defaults';

export const treeModelDefinition: ModelDefinition<'tree'> = {
    key: 'tree',
    label: 'Decision Tree',
    taskTypes: ['regression', 'classification'],
    defaultSettings: (taskType) =>
        taskType === 'regression' ? DEFAULT_REGRESSION_SETTINGS : DEFAULT_CLASSIFICATION_SETTINGS,
    settingsComponent: TreeSettings,

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
        metricsGridComponent: TreeMainMetrics,
        modelDataPlotComponent: TreeModelDataPlots,
        plots: (taskType) =>
            taskType === 'regression'
                ? [
                      { title: 'Metrics', component: RegressionMetrics },
                      { title: 'Residuals', component: ResidualsPlot },
                  ]
                : [{ title: 'Confusion Matrix', component: ConfusionMatrix }],
        parametersComponent: DecisionTreeParameters,
    },

    progress: {
        getProgressInfo: () => {
            return {
                type: 'indeterminate',
                label: '',
            };
        },
    },
};
