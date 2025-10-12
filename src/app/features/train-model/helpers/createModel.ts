import type {
    LinearSettings,
    LogisticSettings,
    ModelSettings,
    NeuralSettings,
    TaskType,
    TreeSettings,
} from '@/app/store';
import type {
    EnsembleTree,
    Model,
    ModelRepresentation,
    TrainingControl,
    TrainingEventEmitter,
} from '@/ml/types';
import {
    BaggingClassifier,
    BaggingRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    LinearRegressor,
    LogisticRegressor,
    NeuralNetwork,
    OneVsRestLogisticRegressor,
    PreprocessingModelDecorator,
    RandomForestClassifier,
    RandomForestRegressor,
    SoftmaxLogisticRegressor,
} from '@/ml/models';
import { EventEmitter } from '@/ml/events/EventEmitter';
import type { Tensor2D } from '@tensorflow/tfjs';
import { calculateOutputFeatures } from '@/ml/data-processing/transformation';
import { TrainingController } from '@/ml/controllers/TrainingController';
import type { TransformationSettings } from '../../transform-data';
import {
    criterionFactory,
    lossFunctionFactory,
    normalizeFunctionFactory,
    regularizationFactory,
    thetaInitializerFactory,
    transformationsFactory,
    optimizerFactory,
} from '@/ml/factories';

export function createModel(
    modelSettings: ModelSettings,
    dataSettings: TransformationSettings,
    taskType: TaskType,
    numFeatures: number,
): [PreprocessingModelDecorator<ModelRepresentation>, TrainingEventEmitter, TrainingControl] {
    try {
        const eventEmitter = new EventEmitter();
        const trainingController = new TrainingController(eventEmitter);
        const model = createBaseModel(
            modelSettings,
            dataSettings,
            taskType,
            numFeatures,
            eventEmitter,
            trainingController,
        );
        const pipeline = createPreprocessingPipeline(model, dataSettings, eventEmitter);

        return [pipeline, eventEmitter, trainingController];
    } catch (error) {
        throw new Error(
            `Failed to create model of type ${modelSettings.type}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        );
    }
}

function createBaseModel(
    modelSettings: ModelSettings,
    dataSettings: TransformationSettings,
    taskType: TaskType,
    numFeatures: number,
    eventEmitter: EventEmitter,
    trainingController: TrainingControl,
): Model<ModelRepresentation> {
    switch (modelSettings.type) {
        case 'tree':
            return createTreeModel(taskType, eventEmitter, modelSettings, trainingController);
        case 'linear':
        case 'logistic':
            return createRegressionModel(modelSettings, eventEmitter, trainingController);
        case 'neural':
            return createNeuralNetworkModel(
                modelSettings,
                dataSettings,
                numFeatures,
                eventEmitter,
                trainingController,
            );
        default:
            throw new Error(`Unsupported model type`);
    }
}

function createPreprocessingPipeline(
    model: Model<ModelRepresentation>,
    dataSettings: TransformationSettings,
    eventEmitter: EventEmitter,
): PreprocessingModelDecorator<ModelRepresentation> {
    const normalizeFunction = normalizeFunctionFactory(dataSettings.normalization);
    const transformations = transformationsFactory(dataSettings.transformations, normalizeFunction);

    const featureTransform = {
        normalizeFunction,
        transformations,
    };

    return new PreprocessingModelDecorator(model, featureTransform, eventEmitter);
}

function createTreeModel(
    taskType: TaskType,
    eventEmitter: EventEmitter,
    modelSettings: TreeSettings,
    trainingController: TrainingControl,
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
    } = modelSettings;
    const isRegression = taskType === 'regression';

    const criterion = criterionFactory(criterionConfig);
    const commonParams = {
        criterion,
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf,
        eventEmitter,
        trainingController,
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

function createRegressionModel(
    modelSettings: LinearSettings | LogisticSettings,
    eventEmitter: EventEmitter,
    trainingController: TrainingControl,
): Model<Tensor2D> {
    const lossFunc = lossFunctionFactory(modelSettings.lossFunction);

    const { type: modelType, optimizer: optimizerConfig } = modelSettings;

    const optimizer = optimizerFactory(optimizerConfig, eventEmitter, trainingController);
    const regularization = regularizationFactory(modelSettings.regularization);
    const thetaInitializer = thetaInitializerFactory(modelSettings.thetaInitialization);

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

function createNeuralNetworkModel(
    modelSettings: NeuralSettings,
    dataSettings: TransformationSettings,
    numFeatures: number,
    eventEmitter: EventEmitter,
    trainingController: TrainingControl,
): Model<Tensor2D> {
    const lossFunc = lossFunctionFactory(modelSettings.lossFunction);

    const { optimizer: optimizerConfig } = modelSettings;

    const optimizer = optimizerFactory(optimizerConfig, eventEmitter, trainingController);
    const regularization = regularizationFactory(modelSettings.regularization);
    const thetaInitializer = thetaInitializerFactory(modelSettings.thetaInitialization);

    const { layers } = modelSettings;
    const unitsOfInputLayer = dataSettings.transformations.reduce((acc, { type, degree }) => {
        return acc + calculateOutputFeatures(type, degree, numFeatures);
    }, numFeatures);
    const layersWithInput = [{ units: unitsOfInputLayer }, ...layers];

    const model = new NeuralNetwork({
        lossFunc,
        optimizer,
        regularization,
        thetaInitializer,
        layers: layersWithInput,
    });

    return model;
}
