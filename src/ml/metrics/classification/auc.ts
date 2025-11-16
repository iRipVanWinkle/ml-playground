/**
 * Computes the Area Under the Curve (AUC) using the trapezoidal rule.
 *
 * @param fpr - The false positive rates.
 * @param tpr - The true positive rates.
 * @returns The AUC as a number.
 */
export function auc(fpr: ArrayLike<number>, tpr: ArrayLike<number>): number {
    if (fpr.length !== tpr.length || fpr.length < 2) {
        return 0;
    }

    let sum = 0;
    for (let i = 1; i < fpr.length; i++) {
        const width = fpr[i] - fpr[i - 1];
        // Skip zero-width intervals (no contribution to AUC)
        if (width === 0) continue;

        // Trapezoidal rule: (width * (height1 + height2)) / 2
        // Factor out division by 2 to the end for better performance
        sum += width * (tpr[i] + tpr[i - 1]);
    }

    return sum / 2;
}
