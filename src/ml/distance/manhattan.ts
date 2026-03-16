import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { ArrayClusteringMath } from './types';

export function manhattanDistance(X: Tensor2D, Y: Tensor2D): Tensor2D {
    return tidy(() => {
        const distances = X.expandDims(1).sub(Y.expandDims(0)).abs().sum(2);

        return distances as Tensor2D;
    });
}

export class ManhattanClusteringMath implements ArrayClusteringMath {
    public distance(a: number[], b: number[]): number {
        let d = 0;
        for (let i = 0; i < a.length; i++) {
            d += Math.abs(a[i] - b[i]);
        }
        return d;
    }

    public centroid(pts: number[][]): number[] {
        if (pts.length === 0) return [];
        const dim = pts[0].length;
        const c = new Array<number>(dim);
        for (let i = 0; i < dim; i++) {
            const values = pts.map((p) => p[i]).sort((a, b) => a - b);
            const mid = Math.floor(values.length / 2);
            c[i] = values.length % 2 !== 0 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
        }
        return c;
    }
}
