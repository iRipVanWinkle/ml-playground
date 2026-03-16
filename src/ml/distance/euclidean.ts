import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { ArrayClusteringMath } from './types';

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

export class EuclideanClusteringMath implements ArrayClusteringMath {
    public distance(a: number[], b: number[]): number {
        let d = 0;
        for (let i = 0; i < a.length; i++) {
            const diff = a[i] - b[i];
            d += diff * diff;
        }
        return Math.sqrt(d);
    }

    public centroid(pts: number[][]): number[] {
        if (pts.length === 0) return [];
        const dim = pts[0].length;
        const c = new Array<number>(dim).fill(0);
        for (const p of pts) {
            for (let i = 0; i < dim; i++) {
                c[i] += p[i];
            }
        }
        for (let i = 0; i < dim; i++) {
            c[i] /= pts.length;
        }
        return c;
    }
}
