import type { DataSettings, ModelSettings } from '@/app/store';
import type { Model, TrainingEventListener } from '@/ml/types';
import { BatchGD, MomentumGD, StochasticGD } from '@/ml/optimizers';
import {
    LinearRegressor,
    LogisticRegressor,
    OneVsRestLogisticRegressor,
    SoftmaxLogisticRegressor,
} from '@/ml/models';
import { ModelPipeline } from '@/ml/ModelPipeline';
import { getLossFunc } from './getLossFunc';
import { getLearningRate } from './getLearningRate';
import { getNormalizeFunc } from './getNormalizeFunc';
import { getRegularization } from './getRegularization';
import { getTransformations } from './getTransformations';

export function createModel(
    modelSettings: ModelSettings,
    dataSettings: DataSettings,
): [Model, TrainingEventListener] {
    const lossFunc = getLossFunc(modelSettings.lossFunction);

    const { type: modelType, optimizer: optimizerConfig } = modelSettings;
    const { scheduler, schedulerConfig, maxIterations, tolerance } = optimizerConfig;

    // Select learning rate
    // If a scheduler is provided, it will return a LearningRate instance; otherwise, it returns a number
    const learningRate = getLearningRate(
        optimizerConfig.learningRate,
        scheduler ? schedulerConfig : undefined,
    );

    const defaultConfig = { learningRate, maxIterations, tolerance };

    // Select optimizer
    let optimizer;
    switch (optimizerConfig.type) {
        case 'momentum': {
            const { beta } = optimizerConfig;
            optimizer = new MomentumGD({ ...defaultConfig, beta });
            break;
        }
        case 'sgd': {
            const { batchSize } = optimizerConfig;
            optimizer = new StochasticGD({ ...defaultConfig, batchSize });
            break;
        }
        case 'batch':
        default:
            optimizer = new BatchGD({ ...defaultConfig });
    }

    // Select regularization
    const regularization = getRegularization(modelSettings.regularization);

    let model;
    switch (modelType) {
        case 'logistic': {
            const { classificationType } = modelSettings;
            if (classificationType === 'softmax') {
                model = new SoftmaxLogisticRegressor({ lossFunc, optimizer, regularization });
            } else if (classificationType === 'ovr') {
                model = new OneVsRestLogisticRegressor({ lossFunc, optimizer, regularization });
            } else {
                model = new LogisticRegressor({ lossFunc, optimizer, regularization });
            }
            break;
        }
        case 'linear':
        default:
            model = new LinearRegressor({ lossFunc, optimizer, regularization });
            break;
    }

    // Select normalization function
    const normalizeFunction = getNormalizeFunc(dataSettings.normalization);

    // Select transformations
    const transformations = getTransformations(dataSettings.transformations, normalizeFunction);

    const featureTransform = {
        normalizeFunction,
        transformations,
    };

    const pipeline = new ModelPipeline(model, featureTransform);

    return [pipeline, optimizer];
}
