export type TaskType = 'regression' | 'classification';
export type ModelType = 'linear' | 'logistic' | 'tree';
export type ClassificationType = 'binary' | 'softmax' | 'ovr';
export type NormalizationFunction = 'none' | 'zscore' | 'linear' | 'log';
export type TransformationFunction = 'sinusoid' | 'cosinusoid' | 'fourier' | 'polynomial';

export type LossFunction =
    | 'mse'
    | 'mae'
    | 'huber'
    | 'logcosh'
    | 'binaryCrossentropy'
    | 'categoricalCrossentropy'
    | 'logitsBasedBinaryCrossentropy'
    | 'logitsBasedCategoricalCrossentropy';
export type CriterionFunction = 'mse' | 'mae' | 'huber' | 'logcosh' | 'gini' | 'entropy';
export type Regularization = 'none' | 'l2';

export type TrainingState = 'init' | 'preparing' | 'training' | 'paused';
export type PendingAction = 'pause' | 'stop' | 'step' | 'resume' | null;

export type TensorBackend = 'auto' | 'webgpu' | 'webgl' | 'cpu' | 'wasm';

export type DataSettings = {
    normalization: NormalizationFunction;
    transformations: Array<{ type: TransformationFunction; degree: number }>;
};

// OPTIMIZATION

type OptimizerBasicConfig = {
    maxIterations: number;
    tolerance: number;
    learningRate: number;
    scheduler: boolean;
    schedulerConfig: { s0: number | undefined; p: number | undefined };
};

type OptimizerBatchConfig = OptimizerBasicConfig & {
    type: 'batch';
};

type OptimizerSGDConfig = OptimizerBasicConfig & {
    type: 'sgd';
    batchSize: number;
};

type OptimizerMomentumConfig = OptimizerBasicConfig & {
    type: 'momentum';
    beta: number;
};

export type OptimizerConfig = OptimizerBatchConfig | OptimizerSGDConfig | OptimizerMomentumConfig;

// LOSS FUNCTION

type LossFunctionGeneralConfig = {
    type: Exclude<LossFunction, 'huber'>;
};

type LossFunctionHuberConfig = {
    type: 'huber';
    delta: number;
};

export type LossFunctionConfig = LossFunctionGeneralConfig | LossFunctionHuberConfig;

// REGULARIZATION

type RegularizationNoneConfig = {
    type: 'none';
};

type RegularizationLConfig = {
    type: 'l2';
    lambda: number;
};

export type RegularizationConfig = RegularizationNoneConfig | RegularizationLConfig;

// THETA INITIALIZATION

type ThetaInitializationBaseConfig = {
    type: 'zeros' | 'ones' | 'xavierUniform' | 'xavierNormal' | 'heUniform' | 'heNormal';
};

type ThetaInitializationConstantConfig = {
    type: 'constant';
    value: number;
};

type ThetaInitializationUniformConfig = {
    type: 'uniform';
    min: number;
    max: number;
};

type ThetaInitializationNormalConfig = {
    type: 'normal';
    mean: number;
    stddev: number;
};

export type ThetaInitializationConfig =
    | ThetaInitializationBaseConfig
    | ThetaInitializationConstantConfig
    | ThetaInitializationUniformConfig
    | ThetaInitializationNormalConfig;

// TREE

export type TreeModelVariant = 'decision' | 'bagging' | 'forest' | 'extra';

type CriterionFunctionGeneralConfig = {
    type: Exclude<CriterionFunction, 'huber'>;
};

type CriterionFunctionHuberConfig = {
    type: 'huber';
    delta: number;
};

export type CriterionFunctionConfig = CriterionFunctionGeneralConfig | CriterionFunctionHuberConfig;

export type TreeSettings = {
    modelVariant: TreeModelVariant;
    criterion: CriterionFunctionConfig;
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    maxFeatures?: number;
    numRandomThresholds?: number;
    estimators?: number;
};

// MODEL SETTINGS

export type ModelSettings = {
    type: ModelType;
    classificationType: ClassificationType;
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
    tree: TreeSettings;
};

export type TrainingReport = {
    trainLossHistory: number[][];
    testLoss: number;
    trainAccuracy: number;
    testAccuracy: number;
    iterations: number[];
    predictionPredictedLabels?: number[][];
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    theta: number[][];
};

export type DataState = {
    trainInputFeatures: number[][];
    trainTargetLabels: number[][];
    testInputFeatures: number[][];
    testTargetLabels: number[][];
    predictionInputFeatures?: number[][];
    xMin: number[];
    xMax: number[];
    headers: string[];
    categories?: string[];
};

export type SystemSettings = {
    backend: TensorBackend;
    randomSeed?: number;
};

export type State = {
    taskType: TaskType;
    dataSettings: DataSettings;
    modelSettings: ModelSettings;
    data: DataState;
    systemSettings: SystemSettings;
    trainingState: TrainingState;
    pendingAction: PendingAction;
    report: TrainingReport;
};
