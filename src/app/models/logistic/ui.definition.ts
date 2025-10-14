import type { ModelDefinition } from '@/app/shared/registry/types';
import { DEFAULT_OPTIMIZER } from '../defaults';
import { LogisticSettings } from './ui/LogisticSettings';

export const logisticModelDefinition: ModelDefinition<'logistic'> = {
    key: 'logistic',
    label: 'Logistic Regression',
    taskTypes: ['classification'],
    defaultSettings: () => ({
        type: 'logistic',
        classificationType: 'binary',
        lossFunction: { type: 'binaryCrossentropy' },
        optimizer: DEFAULT_OPTIMIZER,
        regularization: { type: 'none' },
        thetaInitialization: { type: 'zeros' },
    }),
    settingsComponent: LogisticSettings,
};
