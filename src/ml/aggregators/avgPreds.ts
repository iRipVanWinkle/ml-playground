import { Tensor, tidy, type Tensor2D } from '@tensorflow/tfjs';

export function avgPreds(tensors: Tensor): Tensor2D {
    return tidy(() => tensors.mean(1, true) as Tensor2D);
}
