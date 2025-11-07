export function mcc(confusionMatrix: number[][]): number {
    const numClasses = confusionMatrix.length;

    // Initialize accumulators
    let n = 0;
    let sumDiagonal = 0;
    const rowSums: number[] = new Array(numClasses).fill(0);
    const columnSums: number[] = new Array(numClasses).fill(0);

    // Single pass: calculate n, sumDiagonal, rowSums, and columnSums
    for (let i = 0; i < numClasses; i++) {
        for (let j = 0; j < numClasses; j++) {
            const value = confusionMatrix[i][j];
            n += value;
            rowSums[i] += value;
            columnSums[j] += value;
            if (i === j) {
                sumDiagonal += value;
            }
        }
    }

    if (n === 0) return 0;

    // Single pass: calculate sumRowColProduct, sumRowSumsSquared, and sumColSumsSquared
    let sumRowColProduct = 0;
    let sumRowSumsSquared = 0;
    let sumColSumsSquared = 0;

    for (let k = 0; k < numClasses; k++) {
        const rowSum = rowSums[k];
        const colSum = columnSums[k];
        sumRowColProduct += rowSum * colSum;
        sumRowSumsSquared += rowSum * rowSum;
        sumColSumsSquared += colSum * colSum;
    }

    // Multiclass MCC formula:
    // MCC = (n * sum(diagonal) - sum(rowSums * colSums)) / sqrt((n^2 - sum(rowSums^2)) * (n^2 - sum(colSums^2)))
    const numerator = n * sumDiagonal - sumRowColProduct;
    const sqrtValue = (n * n - sumRowSumsSquared) * (n * n - sumColSumsSquared);

    // Ensure non-negative to prevent sqrt(NaN) from floating point errors
    if (sqrtValue <= 0) return 0;

    const denominator = Math.sqrt(sqrtValue);
    const mcc = numerator / denominator;

    // Clamp MCC to valid range [-1, 1] to handle any floating point errors
    return Math.max(-1, Math.min(1, mcc));
}
