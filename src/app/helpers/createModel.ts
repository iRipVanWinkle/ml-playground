import type { DataSettings, ModelSettings, TaskType } from '@/app/store';
import type { EnsembleTree, Model, ModelRepresentation, TrainingEventEmitter } from '@/ml/types';
import { BatchGD, MomentumGD, StochasticGD } from '@/ml/optimizers';
import {
    BaggingClassifier,
    BaggingRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    LinearRegressor,
    LogisticRegressor,
    OneVsRestLogisticRegressor,
    PreprocessingModelDecorator,
    RandomForestClassifier,
    RandomForestRegressor,
    SoftmaxLogisticRegressor,
} from '@/ml/models';
import { EventEmitter } from '@/ml/events/EventEmitter';
import { getLossFunc } from './getLossFunc';
import { getLearningRate } from './getLearningRate';
import { getNormalizeFunc } from './getNormalizeFunc';
import { getRegularization } from './getRegularization';
import { getTransformations } from './getTransformations';
import { getThetaInitializer } from './getThetaInitializer';
import type { Tensor2D } from '@tensorflow/tfjs';
import { getCriterionFunc } from './getCriterionFunction';

export function createModel(
    modelSettings: ModelSettings,
    dataSettings: DataSettings,
    taskType: TaskType,
): [PreprocessingModelDecorator<ModelRepresentation>, TrainingEventEmitter] {
    try {
        const eventEmitter = new EventEmitter();
        const model = createBaseModel(modelSettings, taskType, eventEmitter);
        const pipeline = createPreprocessingPipeline(model, dataSettings, eventEmitter);

        return [pipeline, eventEmitter];
    } catch (error) {
        throw new Error(
            `Failed to create model of type ${modelSettings.type}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        );
    }
}

function createBaseModel(
    modelSettings: ModelSettings,
    taskType: TaskType,
    eventEmitter: EventEmitter,
): Model<ModelRepresentation> {
    switch (modelSettings.type) {
        case 'tree':
            return createTreeModel(taskType, eventEmitter, modelSettings);
        case 'linear':
        case 'logistic':
            return createRegressionOrNNModel(modelSettings, eventEmitter);
        default:
            throw new Error(`Unsupported model type: ${modelSettings.type}`);
    }
}

function createPreprocessingPipeline(
    model: Model<ModelRepresentation>,
    dataSettings: DataSettings,
    eventEmitter: EventEmitter,
): PreprocessingModelDecorator<ModelRepresentation> {
    const normalizeFunction = getNormalizeFunc(dataSettings.normalization);
    const transformations = getTransformations(dataSettings.transformations, normalizeFunction);

    const featureTransform = {
        normalizeFunction,
        transformations,
    };

    return new PreprocessingModelDecorator(model, featureTransform, eventEmitter);
}

function createTreeModel(
    taskType: TaskType,
    eventEmitter: EventEmitter,
    modelSettings: ModelSettings,
): Model<EnsembleTree> {
    const {
        modelVariant,
        criterion: criterionConfig,
        estimators,
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf,
        maxFeatures,
        numRandomThresholds,
    } = modelSettings.tree;
    const isRegression = taskType === 'regression';

    const criterion = getCriterionFunc(criterionConfig);
    const commonParams = {
        criterion,
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf,
        eventEmitter,
    };
    const ensembleParams = { ...commonParams, estimators };
    const forestParams = { ...ensembleParams, maxFeatures };

    let model;
    switch (modelVariant) {
        case 'decision':
            model = new (isRegression ? DecisionTreeRegressor : DecisionTreeClassifier)(
                commonParams,
            );
            break;
        case 'bagging':
            model = new (isRegression ? BaggingRegressor : BaggingClassifier)(ensembleParams);
            break;
        case 'forest':
            model = new (isRegression ? RandomForestRegressor : RandomForestClassifier)(
                forestParams,
            );
            break;
        case 'extra':
            model = new (isRegression ? ExtraTreesRegressor : ExtraTreesClassifier)({
                ...forestParams,
                numRandomThresholds,
            });
            break;
        default:
            throw new Error(`Unsupported tree model variant: ${modelVariant}`);
    }

    return model;
}

function createRegressionOrNNModel(
    modelSettings: ModelSettings,
    eventEmitter: EventEmitter,
): Model<Tensor2D> {
    const lossFunc = getLossFunc(modelSettings.lossFunction);

    const { type: modelType, optimizer: optimizerConfig } = modelSettings;

    const optimizer = createOptimizer(optimizerConfig, eventEmitter);
    const regularization = getRegularization(modelSettings.regularization);
    const thetaInitializer = getThetaInitializer(modelSettings.thetaInitialization);

    const commonModelParams = {
        lossFunc,
        optimizer,
        regularization,
        thetaInitializer,
    };

    let model;
    switch (modelType) {
        case 'logistic': {
            const { classificationType } = modelSettings;
            if (classificationType === 'softmax') {
                model = new SoftmaxLogisticRegressor(commonModelParams);
            } else if (classificationType === 'ovr') {
                model = new OneVsRestLogisticRegressor(commonModelParams);
            } else {
                model = new LogisticRegressor(commonModelParams);
            }
            break;
        }
        case 'linear':
        default:
            model = new LinearRegressor(commonModelParams);
            break;
    }

    return model;
}

function createOptimizer(optimizerConfig: ModelSettings['optimizer'], eventEmitter: EventEmitter) {
    const { scheduler, schedulerConfig, maxIterations, tolerance } = optimizerConfig;

    const learningRate = getLearningRate(
        optimizerConfig.learningRate,
        scheduler ? schedulerConfig : undefined,
    );

    const baseConfig = {
        learningRate,
        maxIterations,
        tolerance,
        eventEmitter,
    };

    switch (optimizerConfig.type) {
        case 'momentum': {
            const { beta } = optimizerConfig;
            return new MomentumGD({ ...baseConfig, beta });
        }

        case 'sgd': {
            const { batchSize } = optimizerConfig;
            return new StochasticGD({ ...baseConfig, batchSize });
        }

        case 'batch':
            return new BatchGD(baseConfig);
    }
}
