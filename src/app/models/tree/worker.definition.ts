import type { WorkerDefinition } from '@/app/shared/registry';
import { treeModelFactory } from './worker/tree-model-factory';
import { TreeRegressionLiveMetrics } from './worker/tree-regression-metrics';
import { TreeClassificationLiveMetrics } from './worker/tree-classification-metrics';

export const treeWorkerDefinition: WorkerDefinition<'tree'> = {
    key: 'tree',
    modelFactory: treeModelFactory,

    liveMetricsFactory: (model, datasetManager, settings) => {
        if (settings.taskType === 'classification') {
            return TreeClassificationLiveMetrics.factory(model, datasetManager);
        } else {
            return TreeRegressionLiveMetrics.factory(model, datasetManager, settings);
        }
    },

    extractParameters: (report) => {
        if (report.params.length === 0) {
            return null;
        }

        return report.params;
    },
};
