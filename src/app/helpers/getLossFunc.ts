import { MeanAbsoluteError, MeanSquaredError } from '@/ml/losses';
import type { LossFunction } from '@/ml/types';
import type { LossFunctionConfig } from '@/app/store';

export function getLossFunc(lossFunction: LossFunctionConfig): LossFunction {
    switch (lossFunction.type) {
        case 'mae':
            return new MeanAbsoluteError();
        case 'mse':
        default:
            return new MeanSquaredError();
    }
}
