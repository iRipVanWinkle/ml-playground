export type TaskType = 'regression' | 'classification';
export type ModelType = 'linear' | 'logistic';
export type ClassificationType = 'binary' | 'softmax' | 'ovr';
export type NormalizationFunction = 'none' | 'zscore';
export type TransformationFunction = 'sinusoid' | 'polynomial';

export type LossFunction =
    | 'mse'
    | 'mae'
    | 'binaryCrossentropy'
    | 'categoricalCrossentropy'
    | 'logitsBasedBinaryCrossentropy'
    | 'logitsBasedCategoricalCrossentropy';

export type Regularization = 'none' | 'l2';

export type TrainingState = 'init' | 'training' | 'paused';

export type DataSettings = {
    normalization: NormalizationFunction;
    transformations: Array<{ type: TransformationFunction; degree: number }>;
};

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

type LossFunctionGeneralConfig = {
    type: LossFunction;
};

export type LossFunctionConfig = LossFunctionGeneralConfig;

type RegularizationNoneConfig = {
    type: 'none';
};

type RegularizationLConfig = {
    type: 'l2';
    lambda: number;
};

export type RegularizationConfig = RegularizationNoneConfig | RegularizationLConfig;

export type ModelSettings = {
    type: ModelType;
    classificationType: ClassificationType;
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
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

export type State = {
    taskType: TaskType;
    dataSettings: DataSettings;
    modelSettings: ModelSettings;
    data: DataState;
    trainingState: TrainingState;
    report: TrainingReport;
};
