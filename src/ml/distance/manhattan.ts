import { tidy, type Tensor2D } from '@tensorflow/tfjs';

export function manhattanDistance(X: Tensor2D, Y: Tensor2D): Tensor2D {
    return tidy(() => {
        const distances = X.expandDims(1).sub(Y.expandDims(0)).abs().sum(2);

        return distances as Tensor2D;
    });
}
