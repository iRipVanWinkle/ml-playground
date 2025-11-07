/**
 * Calculate macro average (unweighted mean)
 */
export function macroAverage(values: number[]): number {
    if (values.length === 0) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return sum / values.length;
}

/**
 * Calculate weighted average using row sums (actual class frequencies) as weights
 */
export function weightedAverage(
    values: number[],
    confusionMatrix: number[][],
    useRowSums: boolean = true,
): number {
    const numClasses = confusionMatrix.length;
    let totalWeight = 0;
    let weightedSum = 0;

    for (let i = 0; i < numClasses; i++) {
        let weight = 0;

        if (useRowSums) {
            // Use row sums (actual class frequencies) for recall/F1 weighting
            // i = row = expected/actual class
            for (let j = 0; j < numClasses; j++) {
                weight += confusionMatrix[i][j];
            }
        } else {
            // Use column sums (predicted class frequencies) for precision weighting
            // i = col = predicted class
            for (let j = 0; j < numClasses; j++) {
                weight += confusionMatrix[j][i];
            }
        }

        totalWeight += weight;
        weightedSum += values[i] * weight;
    }

    return totalWeight > 0 ? weightedSum / totalWeight : 0;
}
