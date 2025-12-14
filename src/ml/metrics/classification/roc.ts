import { Matrix, type MatrixLike } from '../../matrix';
/**
 * A type representing a ROC curve.
 *
 * @param fpr - The false positive rates.
 * @param tpr - The true positive rates.
 * @param thresholds - The thresholds.
 * @param youdenOptimalIndex - The index of the optimal threshold (based on Youden's J statistic).
 * @param closestToCornerIndex - The index of the optimal threshold (closest point to the (0,1) corner).
 */
export type RocCurve = {
    fpr: Float32Array;
    tpr: Float32Array;
    thresholds: Float32Array;
    youdenOptimalIndex: number | null;
    closestToCornerIndex: number | null;
};

/**
 * A type representing a multiclass ROC curve.
 *
 * @param curves - The ROC curves for each class.
 * @param classIndices - The class indices.
 */
export type MulticlassRocCurve = {
    curves: RocCurve[];
    classIndices: number[];
};

/**
 * Computes the ROC curve for a binary classification problem.
 *
 * @param yTrue - The true labels.
 * @param yProb - The predicted probabilities.
 * @returns The ROC curve as an object containing fpr, tpr, and thresholds.
 */
export function rocCurve(yTrue: ArrayLike<number>, yProb: ArrayLike<number>): RocCurve {
    const numSamples = yProb.length;

    // Create array of indices sorted by probability (descending)
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    indices.sort((a, b) => yProb[b] - yProb[a]);

    // Count positives and negatives
    let totalPositives = 0;
    let totalNegatives = 0;

    for (let i = 0; i < numSamples; i++) {
        if (yTrue[i] === 1) {
            totalPositives++;
        } else {
            totalNegatives++;
        }
    }

    // Initialize arrays
    const fpr: number[] = [0];
    const tpr: number[] = [0];
    const thresholds: number[] = [1.0 + 1e-8]; // Start above max probability

    let truePositives = 0;
    let falsePositives = 0;

    let maxJ = -Infinity;
    let optimalIdx = null;
    let minDistance = Infinity;
    let optimalIdxDistance = null;

    // Process each threshold
    for (let i = 0; i < numSamples; i++) {
        const idx = indices[i];
        const prob = yProb[idx];
        const label = yTrue[idx];

        if (label === 1) {
            truePositives++;
        } else {
            falsePositives++;
        }

        // Calculate TPR and FPR
        const currentTPR = totalPositives > 0 ? truePositives / totalPositives : 0;
        const currentFPR = totalNegatives > 0 ? falsePositives / totalNegatives : 0;

        tpr.push(currentTPR);
        fpr.push(currentFPR);
        thresholds.push(prob);

        // Find the optimal threshold based on Youden's J statistic
        const j = tpr[i] - fpr[i];
        if (j > maxJ) {
            maxJ = j;
            optimalIdx = i;
        }

        // Find the optimal threshold based on the distance to the (0,1) corner
        const distance = Math.sqrt(fpr[i] ** 2 + (1 - tpr[i]) ** 2);
        if (distance < minDistance) {
            minDistance = distance;
            optimalIdxDistance = i;
        }
    }

    // Ensure we end at (1, 1)
    if (tpr[tpr.length - 1] !== 1 || fpr[fpr.length - 1] !== 1) {
        tpr.push(1);
        fpr.push(1);
        thresholds.push(0);
    }

    const curve: RocCurve = {
        fpr: Float32Array.from(fpr),
        tpr: Float32Array.from(tpr),
        thresholds: Float32Array.from(thresholds),
        youdenOptimalIndex: optimalIdx,
        closestToCornerIndex: optimalIdxDistance,
    };

    return curve;
}

/**
 * Computes the ROC curve for a multiclass classification problem.
 * The ROC curve is computed for each class separately.
 *
 * @param yTrue - The true labels.
 * @param yProb - The predicted probabilities.
 * @returns The ROC curve as an object containing curves and class indices.
 * The curves are sorted by the class indices.
 */
export function multiclassRocCurve(yTrue: MatrixLike, yProb: MatrixLike): MulticlassRocCurve {
    const numExamples = yProb.shape[0];
    const numClasses = yProb.shape[1];

    const curves: RocCurve[] = [];
    const classIndices: number[] = [];

    const yTrueFlat = yTrue.array;
    const probMatrix = Matrix.from(yProb);

    for (let classIdx = 0; classIdx < numClasses; classIdx++) {
        // Extract probabilities for this class
        const probFlat = probMatrix.col(classIdx);
        const binaryLabels = new Uint8Array(yTrue.shape[0]);

        for (let i = 0; i < numExamples; i++) {
            binaryLabels[i] = yTrueFlat[i] === classIdx ? 1 : 0;
        }

        const curve = rocCurve(binaryLabels, probFlat);

        curves.push(curve);
        classIndices.push(classIdx);
    }

    return {
        curves,
        classIndices,
    };
}
