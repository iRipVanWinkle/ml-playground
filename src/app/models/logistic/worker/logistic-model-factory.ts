import {
    lossFunctionFactory,
    optimizerFactory,
    regularizationFactory,
    thetaInitializerFactory,
} from '@/ml/factories';
import {
    LogisticRegressor,
    OneVsRestLogisticRegressor,
    SoftmaxLogisticRegressor,
} from '@/ml/models';
import type { TrainingSettings } from '../../types';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { LogisticSettings } from '../types';

export function logisticModelFactory(
    settings: TrainingSettings<LogisticSettings>,
    eventEmitter: TrainingEventEmitter,
    trainingController: TrainingControl,
) {
    const { modelSettings } = settings;
    const lossFunc = lossFunctionFactory(modelSettings.lossFunction);

    const { optimizer: optimizerConfig, classificationType } = modelSettings;

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
    switch (classificationType) {
        case 'softmax':
            model = new SoftmaxLogisticRegressor(commonModelParams);
            break;
        case 'ovr':
            model = new OneVsRestLogisticRegressor(commonModelParams);
            break;
        case 'binary':
        default:
            model = new LogisticRegressor(commonModelParams);
            break;
    }

    return model;
}
