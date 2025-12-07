import type { ModelDefinition } from '@/app/shared/registry/types';
import type { TaskType } from '@/app/shared/types';
import { EMPTY_MATRIX_LIKE } from '@/ml/matrix';
import { TreeSettings } from './ui/TreeSettings';
import { TreeMainMetrics } from './ui/TreeMainMetrics';
import { TreeModelDataPlots } from './ui/TreeModelDataPlots';
import {
    ConfusionMatrix,
    RegressionMetrics,
    ResidualsPlot,
    RocCurve,
} from '@/app/shared/visualization';

export const treeModelDefinition: ModelDefinition<'tree'> = {
    key: 'tree',
    label: 'Decision Tree',
    taskTypes: ['regression', 'classification'],
    defaultSettings: (taskType?: TaskType) => ({
        type: 'tree',
        modelVariant: 'decision',
        criterion: { type: taskType === 'regression' ? 'mse' : 'gini' },
        maxDepth: 5,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
        estimators: 10,
        maxFeatures: 1,
        numRandomThresholds: 1,
    }),
    settingsComponent: TreeSettings,

    defaultReport: (taskType: TaskType) => {
        switch (taskType) {
            case 'classification':
                return {
                    type: 'tree',
                    taskType: 'classification',
                    iterations: [],
                    testAccuracy: 0,
                    trainAccuracy: 0,
                    trainPredictedLabels: EMPTY_MATRIX_LIKE,
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
                    type: 'tree',
                    taskType: 'regression',
                    iterations: [],
                    testLoss: 0,
                    trainPredictedLabels: EMPTY_MATRIX_LIKE,
                    trainMetrics: null,
                    trainResiduals: EMPTY_MATRIX_LIKE,
                };
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
                : [
                      { title: 'Confusion Matrix', component: ConfusionMatrix },
                      { title: 'ROC Curve', component: RocCurve },
                  ],
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
