import type { ModelSettings, ModelType, TaskType } from './types';

const DEFAULT_OPTIMIZER = {
    type: 'batch',
    maxIterations: 100,
    tolerance: 0.0001,
    learningRate: 0.01,
    scheduler: false,
    schedulerConfig: { s0: 1, p: 0.5 },
} as const;

export const modelSettingsDefaults: Record<ModelType, (taskType: TaskType) => ModelSettings> = {
    linear: () => ({
        type: 'linear',
        lossFunction: { type: 'mse' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'zeros' },
        layers: [{ units: 1, activation: 'linear' }],
    }),
    logistic: () => ({
        type: 'logistic',
        classificationType: 'binary',
        lossFunction: { type: 'binaryCrossentropy' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'zeros' },
    }),
    neural: (taskType) => ({
        type: 'neural',
        lossFunction: { type: taskType === 'regression' ? 'mse' : 'binaryCrossentropy' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'xavierNormal' },
        layers: [{ units: 1, activation: 'linear' }],
    }),
    tree: (taskType) => ({
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
};
