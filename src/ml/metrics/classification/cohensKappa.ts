/**
 * Computes the Cohen's Kappa metric from a confusion matrix.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The Cohen's Kappa as a number.
 */
export function cohensKappa(confusionMatrix: number[][]): number {
    const numClasses = confusionMatrix.length;

    let sumTotal = 0;
    let sumRowColProduct = 0;
    let sumDiagonal = 0;

    // Calculate row sums (actual class frequencies) and column sums (predicted class frequencies)
    // in a single pass, while also computing total n and diagonal sum
    for (let i = 0; i < numClasses; i++) {
        let rowSum = 0;
        let colSum = 0;

        for (let j = 0; j < numClasses; j++) {
            sumTotal += confusionMatrix[i][j]; // Accumulate total while iterating
            rowSum += confusionMatrix[i][j];
            colSum += confusionMatrix[j][i];
        }

        sumRowColProduct += rowSum * colSum;
        sumDiagonal += confusionMatrix[i][i];
    }

    if (sumTotal === 0) return 0;
    // Calculate observed agreement (p₀) = accuracy = sum of diagonal / total
    const p0 = sumDiagonal / sumTotal;

    // Calculate expected agreement by chance (pₑ) = sum(rowSum[i] * colSum[i]) / n²
    const pe = sumRowColProduct / (sumTotal * sumTotal);

    // Calculate Cohen's Kappa: κ = (p₀ - pₑ) / (1 - pₑ)
    const denominator = 1 - pe;
    const kappa = denominator > 0 ? (p0 - pe) / denominator : 0;

    // Clamp Kappa to valid range [-1, 1] to handle any floating point errors
    return Math.max(-1, Math.min(1, kappa));
}
