import { Entropy, Gini, Huber, LogCosh, MeanAbsoluteError, MeanSquaredError } from '../../criteria';
import type { CriterionFunction } from '../../types';
import type { CriterionConfig } from './types';

export function criterionFactory(criterion: CriterionConfig): CriterionFunction {
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
