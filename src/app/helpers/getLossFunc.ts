import {
    MeanAbsoluteError,
    MeanSquaredError,
    BinaryCrossentropy,
    BinaryCrossentropyLogits,
    CategoricalCrossentropy,
    CategoricalCrossentropyLogits,
} from '@/ml/losses';
import type { LossFunction } from '@/ml/types';
import type { LossFunctionConfig } from '@/app/store';

export function getLossFunc(lossFunction: LossFunctionConfig): LossFunction {
    switch (lossFunction.type) {
        case 'binaryCrossentropy':
            return new BinaryCrossentropy();
        case 'logitsBasedBinaryCrossentropy':
            return new BinaryCrossentropyLogits();
        case 'categoricalCrossentropy':
            return new CategoricalCrossentropy();
        case 'logitsBasedCategoricalCrossentropy':
            return new CategoricalCrossentropyLogits();
        case 'mae':
            return new MeanAbsoluteError();
        case 'mse':
        default:
            return new MeanSquaredError();
    }
}
