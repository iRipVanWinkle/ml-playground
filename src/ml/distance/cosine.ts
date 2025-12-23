import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import { EPSILON } from '../constants';

export function cosineDistance(X: Tensor2D, Y: Tensor2D): Tensor2D {
    return tidy(() => {
        const xNorm = X.norm(2, 1, true);
        const cNorm = Y.norm(2, 1).expandDims(0);

        const normalizedX = X.div(xNorm.add(EPSILON));
        const normalizedY = Y.div(cNorm.transpose().add(EPSILON));

        const cosineSimilarity = normalizedX.matMul(normalizedY.transpose());

        return cosineSimilarity.mul(-1).add(1);
    });
}
