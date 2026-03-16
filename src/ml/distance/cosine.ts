import { tidy, type Tensor2D } from '@tensorflow/tfjs';
import { EPSILON } from '../constants';
import type { ArrayClusteringMath } from './types';

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

export class CosineClusteringMath implements ArrayClusteringMath {
    public distance(a: number[], b: number[]): number {
        let dotProduct = 0;
        let aNormSq = 0;
        let bNormSq = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            aNormSq += a[i] * a[i];
            bNormSq += b[i] * b[i];
        }
        const aNorm = Math.sqrt(aNormSq);
        const bNorm = Math.sqrt(bNormSq);
        return 1 - dotProduct / (aNorm * bNorm + EPSILON);
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

        let normSq = 0;
        for (let i = 0; i < c.length; i++) {
            normSq += c[i] * c[i];
        }
        const norm = Math.sqrt(normSq);
        if (norm === 0) return c;

        for (let i = 0; i < c.length; i++) {
            c[i] /= norm;
        }
        return c;
    }
}
