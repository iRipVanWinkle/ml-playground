import {
    lossFunctionFactory,
    optimizerFactory,
    regularizationFactory,
    thetaInitializerFactory,
} from '@/ml/factories';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import { NeuralNetwork } from '@/ml/models';
import { calculateOutputFeatures } from '@/app/shared/helpers';
import type { TrainingSettings } from '../../types';
import type { NeuralSettings } from '../types';

export function neuralModelFactory(
    settings: TrainingSettings<NeuralSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { modelSettings, dataSettings, dataset } = settings;
    const lossFunc = lossFunctionFactory(modelSettings.lossFunction);

    const { optimizer: optimizerConfig, layers } = modelSettings;

    const optimizer = optimizerFactory(optimizerConfig, eventEmitter, trainingController);
    const regularization = regularizationFactory(modelSettings.regularization);
    const thetaInitializer = thetaInitializerFactory(modelSettings.thetaInitialization);

    const numFeatures = dataset.trainInputFeatures[0].length;
    const unitsOfInputLayer = dataSettings.transformations.reduce((acc, { type, degree }) => {
        return acc + calculateOutputFeatures(type, degree, numFeatures);
    }, numFeatures);
    const layersWithInput = [{ units: unitsOfInputLayer }, ...layers];

    return new NeuralNetwork({
        lossFunc,
        optimizer,
        regularization,
        thetaInitializer,
        layers: layersWithInput,
    });
}
