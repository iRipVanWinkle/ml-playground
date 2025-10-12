import {
    MeanAbsoluteError,
    MeanSquaredError,
    BinaryCrossentropy,
    BinaryCrossentropyLogits,
    CategoricalCrossentropy,
    CategoricalCrossentropyLogits,
    Huber,
    LogCosh,
} from '../../losses';
import type { LossFunction } from '../../types';
import type { LossFunctionConfig } from './types';

export function lossFunctionFactory(lossFunction: LossFunctionConfig): LossFunction {
    switch (lossFunction.type) {
        case 'binaryCrossentropy':
            return new BinaryCrossentropy();
        case 'logitsBasedBinaryCrossentropy':
            return new BinaryCrossentropyLogits();
        case 'categoricalCrossentropy':
            return new CategoricalCrossentropy();
        case 'logitsBasedCategoricalCrossentropy':
            return new CategoricalCrossentropyLogits();
        case 'logcosh':
            return new LogCosh();
        case 'huber':
            return new Huber(lossFunction.delta);
        case 'mae':
            return new MeanAbsoluteError();

        case 'mse':
        default:
            return new MeanSquaredError();
    }
}
