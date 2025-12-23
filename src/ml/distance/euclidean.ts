import { tidy, type Tensor2D } from '@tensorflow/tfjs';

export function euclideanDistance(X: Tensor2D, Y: Tensor2D): Tensor2D {
    return tidy(() => {
        const xNorm = X.square().sum(1, true);
        const cNorm = Y.square().sum(1).expandDims(0);
        const cross = X.matMul(Y.transpose()).mul(-2);

        const squaredDistances = xNorm.add(cNorm).add(cross);

        // stabilize distances to avoid NaNs from sqrt of negative numbers
        const stabilizedDistances = squaredDistances.maximum(0);

        return stabilizedDistances.sqrt() as Tensor2D;
    });
}
