import {
    lossFunctionFactory,
    optimizerFactory,
    regularizationFactory,
    thetaInitializerFactory,
} from '@/ml/factories';
import { LinearRegressor } from '@/ml/models';
import type { TrainingSettings } from '../../types';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { LinearSettings } from '../types';

export function linearModelFactory(
    settings: TrainingSettings<LinearSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { modelSettings } = settings;
    const lossFunc = lossFunctionFactory(modelSettings.lossFunction);

    const { optimizer: optimizerConfig } = modelSettings;

    const optimizer = optimizerFactory(optimizerConfig, eventEmitter, trainingController);
    const regularization = regularizationFactory(modelSettings.regularization);
    const thetaInitializer = thetaInitializerFactory(modelSettings.thetaInitialization);

    const commonModelParams = {
        lossFunc,
        optimizer,
        regularization,
        thetaInitializer,
    };

    return new LinearRegressor(commonModelParams);
}
