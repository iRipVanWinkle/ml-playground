import type { ModelDefinition } from '@/app/shared/registry/types';
import { NaiveBayesSettings } from './ui/NaiveBayesSettings';
import { NaiveBayesMainMetrics } from './ui/NaiveBayesMainMetrics';
import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import {
    LogisticPlots,
    ClassConditionalPlot,
    ConfusionMatrix,
    RocCurve,
    NaiveBayesParameters,
} from '@/app/shared/visualization';

export const naiveBayesModelDefinition: ModelDefinition<'naive-bayes'> = {
    key: 'naive-bayes',
    label: 'Naive Bayes',
    taskTypes: ['classification'],
    defaultSettings: () => ({
        type: 'naive-bayes',
        variant: 'gaussian',
    }),
    settingsComponent: NaiveBayesSettings,

    defaultReport: () => ({
        type: 'naive-bayes',
        taskType: 'classification',
        testAccuracy: 0,
        trainAccuracy: 0,
        trainPredictedLabels: EMPTY_MATRIX_LIKE,
        iteration: 0,
        params: {
            type: 'gaussian',
            classes: [],
            classPriors: new Float32Array(),
            classMeans: EMPTY_MATRIX_LIKE,
            classVariances: EMPTY_MATRIX_LIKE,
        },
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
    }),
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
