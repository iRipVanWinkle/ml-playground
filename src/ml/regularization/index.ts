import { scalar, zeros, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Regularization } from '../types';

export class NoRegularization implements Regularization {
    zero = scalar(0);

    compute(): Scalar {
        return this.zero;
    }

    gradient(theta: Tensor2D): Tensor2D {
        return zeros(theta.shape);
    }
}

export * from './l2';
