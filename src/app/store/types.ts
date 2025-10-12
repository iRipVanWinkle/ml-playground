import type {
    CriterionConfig,
    LossFunctionConfig,
    OptimizerConfig,
    RegularizationConfig,
    ThetaInitializationConfig,
} from '@/ml/factories';

export type TaskType = 'regression' | 'classification';
export type ClassificationType = 'binary' | 'softmax' | 'ovr';

export type TrainingState = 'init' | 'preparing' | 'training' | 'paused';
export type PendingAction = 'pause' | 'stop' | 'step' | 'resume' | null;

// TREE

export type TreeModelVariant = 'decision' | 'bagging' | 'forest' | 'extra';

// MODEL SETTINGS

export type LinearSettings = {
    type: 'linear';
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
};

export type LogisticSettings = {
    type: 'logistic';
    classificationType: ClassificationType;
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
};

export type NeuralSettings = {
    type: 'neural';
    lossFunction: LossFunctionConfig;
    optimizer: OptimizerConfig;
    regularization: RegularizationConfig;
    thetaInitialization: ThetaInitializationConfig;
    layers: Array<{ units: number; activation?: string }>;
};

export type TreeSettings = {
    type: 'tree';
    modelVariant: TreeModelVariant;
    criterion: CriterionConfig;
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    maxFeatures?: number;
    numRandomThresholds?: number;
    estimators?: number;
};

export type ModelSettings = LinearSettings | LogisticSettings | NeuralSettings | TreeSettings;

export type ModelType = ModelSettings['type'];

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

export type State = {
    taskType: TaskType;
    modelSettings: ModelSettings;
    trainingState: TrainingState;
    pendingAction: PendingAction;
    report: TrainingReport;
};
