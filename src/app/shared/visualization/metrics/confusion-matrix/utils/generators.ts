/**
 * Generates labels for a confusion matrix
 * @param matrixSize - The size of the matrix
 * @returns An array of labels
 */
export function generateLabels(matrixSize: number): string[] {
    return Array.from({ length: matrixSize }, (_, i) => `${i + 1}`);
}
