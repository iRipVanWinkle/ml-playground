import type { Tensor, Tensor1D, Tensor2D } from '@tensorflow/tfjs';

/**
 * Type guard to check if a Tensor is Tensor2D
 */
export function isTensor2D(obj: Tensor): obj is Tensor2D {
    return obj.rank === 2;
}

/**
 * Type guard to check if a Tensor is Tensor1D
 */
export function isTensor1D(obj: Tensor): obj is Tensor1D {
    return obj.rank === 1;
}
