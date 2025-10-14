import type { ModelDefinition } from '@/app/shared/registry/types';
import type { TaskType } from '@/app/shared/types';
import { NeuralSettings } from './ui/NeuralSettings';
import { DEFAULT_OPTIMIZER } from '../defaults';

export const neuralModelDefinition: ModelDefinition<'neural'> = {
    key: 'neural',
    label: 'Neural Networks',
    taskTypes: ['regression', 'classification'],
    defaultSettings: (taskType?: TaskType) => ({
        type: 'neural',
        lossFunction: { type: taskType === 'regression' ? 'mse' : 'binaryCrossentropy' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'xavierNormal' },
        layers: [{ units: 1, activation: 'linear' }],
    }),
    settingsComponent: NeuralSettings,
};
