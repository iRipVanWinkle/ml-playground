import type { ModelDefinition } from '@/app/shared/registry/types';
import type { TaskType } from '@/app/shared/types';
import { TreeSettings } from './ui/TreeSettings';
import { TreeMainMetrics } from './ui/TreeMainMetrics';
import { TreeModelDataPlots } from './ui/TreeModelDataPlots';

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

    visualization: {
        metricsGridComponent: TreeMainMetrics,
        modelDataPlotComponent: TreeModelDataPlots,
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
