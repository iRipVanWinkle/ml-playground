import type { WorkerDefinition } from '@/app/shared/registry';
import { neuralModelFactory } from './worker/neural-model-factory';
import { NeuralRegressionLiveMetrics } from './worker/neural-regression-metrics';
import { NeuralClassificationLiveMetrics } from './worker/neural-classification-metrics';

export const neuralWorkerDefinition: WorkerDefinition<'neural'> = {
    key: 'neural',
    modelFactory: neuralModelFactory,
    liveMetricsFactory: (model, datasetManager, taskType) => {
        if (taskType === 'classification') {
            return NeuralClassificationLiveMetrics.factory(model, datasetManager);
        } else {
            return NeuralRegressionLiveMetrics.factory(model, datasetManager);
        }
    },
};
