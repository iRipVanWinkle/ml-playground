import { Entropy, Gini, Huber, LogCosh, MeanAbsoluteError, MeanSquaredError } from '@/ml/criteria';
import type { CriterionFunction } from '@/ml/types';
import type { CriterionFunctionConfig } from '@/app/store';

export function getCriterionFunc(criterion: CriterionFunctionConfig): CriterionFunction {
    switch (criterion.type) {
        case 'mae':
            return new MeanAbsoluteError();
        case 'huber':
            return new Huber(criterion.delta);
        case 'logcosh':
            return new LogCosh();
        case 'gini':
            return new Gini();
        case 'entropy':
            return new Entropy();
        case 'mse':
        default:
            return new MeanSquaredError();
    }
}
