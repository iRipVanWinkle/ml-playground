import type { ModelDefinition } from '@/app/shared/registry/types';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { LinearSettings } from './ui/LinearSettings';

export const linearModelDefinition: ModelDefinition<'linear'> = {
    key: 'linear',
    label: 'Linear Regression',
    taskTypes: ['regression'],
    defaultSettings: () => ({
        type: 'linear',
        lossFunction: { type: 'mse' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'zeros' },
    }),
    settingsComponent: LinearSettings,
};
