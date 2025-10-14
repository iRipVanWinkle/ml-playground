import {
    lossFunctionFactory,
    optimizerFactory,
    regularizationFactory,
    thetaInitializerFactory,
} from '@/ml/factories';
import { NeuralNetwork } from '@/ml/models';
import type { TrainingSettings } from '../../types';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { NeuralSettings } from '../types';
import { calculateOutputFeatures } from '@/ml/data-processing/transformation';

export function neuralModelFactory(
    settings: TrainingSettings<NeuralSettings>,
    eventEmitter: TrainingEventEmitter,
    trainingController: TrainingControl,
) {
    const { modelSettings, dataSettings, data } = settings;
    const lossFunc = lossFunctionFactory(modelSettings.lossFunction);

    const { optimizer: optimizerConfig, layers } = modelSettings;

    const optimizer = optimizerFactory(optimizerConfig, eventEmitter, trainingController);
    const regularization = regularizationFactory(modelSettings.regularization);
    const thetaInitializer = thetaInitializerFactory(modelSettings.thetaInitialization);

    const numFeatures = data.trainInputFeatures[0].length;
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
