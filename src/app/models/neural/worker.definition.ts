import type { WorkerDefinition } from '@/app/shared/registry';
import { neuralModelFactory } from './worker/neural-model-factory';
import { NeuralRegressionLiveMetrics } from './worker/neural-regression-metrics';
import { NeuralClassificationLiveMetrics } from './worker/neural-classification-metrics';
import { tensor2d } from '@tensorflow/tfjs';

export const neuralWorkerDefinition: WorkerDefinition<'neural'> = {
    key: 'neural',
    modelFactory: neuralModelFactory,
    liveMetricsFactory: (model, datasetManager, settings) => {
        if (settings.taskType === 'classification') {
            return NeuralClassificationLiveMetrics.factory(model, datasetManager);
        } else {
            return NeuralRegressionLiveMetrics.factory(model, datasetManager, settings);
        }
    },

    extractParameters: (report) => {
        const { theta } = report;
        return tensor2d(theta.array, theta.shape);
    },
};
