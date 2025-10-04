import {
    ElasticNetRegularization,
    L1Regularization,
    L2Regularization,
    NoRegularization,
} from '@/ml/regularization';
import type { Regularization } from '@/ml/types';
import type { RegularizationConfig } from '@/app/store';

export function getRegularization(regularization: RegularizationConfig): Regularization {
    switch (regularization.type) {
        case 'l2':
            return new L2Regularization(regularization.lambda);
        case 'l1':
            return new L1Regularization(regularization.lambda);
        case 'elasticnet':
            return new ElasticNetRegularization(regularization.lambda, regularization.alpha);
        default:
            return new NoRegularization();
    }
}
