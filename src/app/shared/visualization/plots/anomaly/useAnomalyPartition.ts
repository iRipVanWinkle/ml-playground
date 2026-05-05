import type { MatrixLike } from '@/app/shared/helpers';

type Point2D = { x: number[]; y: number[] };

type AnomalyPartition = {
    normal: Point2D;
    anomaly: Point2D;
    hasPredictions: boolean;
};

export function useAnomalyPartition(
    inputFeatures: number[][],
    predictions: MatrixLike | undefined,
): AnomalyPartition {
    const normal: Point2D = { x: [], y: [] };
    const anomaly: Point2D = { x: [], y: [] };
    const hasPredictions = (predictions?.array.length ?? 0) > 0;

    if (hasPredictions) {
        for (let i = 0; i < inputFeatures.length; i++) {
            // -1 = anomaly, >= 0 normal
            const target = predictions!.array[i] < 0 ? anomaly : normal;
            target.x.push(inputFeatures[i][0]);
            target.y.push(inputFeatures[i][1]);
        }
    } else {
        for (const p of inputFeatures) {
            normal.x.push(p[0]);
            normal.y.push(p[1]);
        }
    }

    return { normal, anomaly, hasPredictions };
}
