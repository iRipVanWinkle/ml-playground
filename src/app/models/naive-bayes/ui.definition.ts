import type { ModelDefinition } from '@/app/shared/registry/types';
import {
    LogisticPlots,
    ClassConditionalPlot,
    ConfusionMatrix,
    RocCurve,
    NaiveBayesParameters,
} from '@/app/shared/visualization';
import { NaiveBayesSettings } from './ui/NaiveBayesSettings';
import { NaiveBayesMainMetrics } from './ui/NaiveBayesMainMetrics';
import { DEFAULT_REPORT, DEFAULT_SETTINGS } from './defaults';

export const naiveBayesModelDefinition: ModelDefinition<'naive-bayes'> = {
    key: 'naive-bayes',
    label: 'Naive Bayes',
    taskTypes: ['classification'],

    defaultSettings: () => DEFAULT_SETTINGS,
    settingsComponent: NaiveBayesSettings,

    defaultReport: () => DEFAULT_REPORT,
    visualization: {
        metricsGridComponent: NaiveBayesMainMetrics,
        modelDataPlotComponent: LogisticPlots,
        plots: [
            { title: 'Conditional Distributions', component: ClassConditionalPlot },
            { title: 'Confusion Matrix', component: ConfusionMatrix },
            { title: 'ROC Curve', component: RocCurve },
        ],
        parametersComponent: NaiveBayesParameters,
    },

    progress: {
        getProgressInfo: ({ report, dataset }) => {
            const current = report.iteration ?? 0;
            const max = dataset.categories?.length ?? 0;

            return {
                type: 'determinate',
                label: `${current}/${max}`,
                current,
                max,
            };
        },
    },
};
