import { L2Regularization, NoRegularization } from '@/ml/regularization';
import type { Regularization } from '@/ml/types';
import type { RegularizationConfig } from '@/app/store';

export function getRegularization(regularization: RegularizationConfig): Regularization {
    switch (regularization.type) {
        case 'l2':
            return new L2Regularization(regularization.lambda);
        default:
            return new NoRegularization();
    }
}
