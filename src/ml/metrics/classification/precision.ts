import { macroAverage, weightedAverage } from './utils';

/**
 * Computes the precision metric for each class from a confusion matrix.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The precision for each class as an array.
 */
export function precision(confusionMatrix: number[][]): number[] {
    const numClasses = confusionMatrix.length;
    const precisions: number[] = [];

    // j = col = predicted class
    for (let j = 0; j < numClasses; j++) {
        let predictedPositives = 0;
        let truePositives = 0;

        // i = row = expected/actual class
        for (let i = 0; i < numClasses; i++) {
            predictedPositives += confusionMatrix[i][j];
            if (i === j) {
                truePositives = confusionMatrix[i][j];
            }
        }

        // Precision = TP / all predicted positives
        precisions[j] = predictedPositives > 0 ? truePositives / predictedPositives : 0;
    }

    return precisions;
}

/**
 * Computes the macro-averaged precision from precision values.
 *
 * Macro average calculates the unweighted mean of precision scores across all classes,
 * treating each class equally regardless of its frequency in the dataset.
 *
 * @param precision - The precision values for each class as an array.
 * @returns The macro-averaged precision as a single number.
 */
export function macroPrecision(precision: number[]): number;

/**
 * Computes the macro-averaged precision from a confusion matrix.
 *
 * Macro average calculates the unweighted mean of precision scores across all classes,
 * treating each class equally regardless of its frequency in the dataset.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The macro-averaged precision as a single number.
 */
export function macroPrecision(confusionMatrix: number[][]): number;

/**
 * Computes the macro-averaged precision.
 *
 * The macro average is calculated by:
 * 1. Computing precision for each class
 * 2. Taking the unweighted mean of all precision scores
 *
 * This metric treats all classes equally, making it useful when you want to
 * evaluate performance across all classes without bias toward more frequent classes.
 *
 * @param confusionMatrixOrPrecision - The confusion matrix or precision values.
 * @returns The macro-averaged precision as a single number.
 */
export function macroPrecision(confusionMatrixOrPrecision: number[][] | number[]): number {
    let precisionArr: number[];

    // Check if first parameter is a confusion matrix (2D array) or precision values (1D array)
    if (Array.isArray(confusionMatrixOrPrecision[0])) {
        // It's a confusion matrix - compute precision from it
        const confusionMatrix = confusionMatrixOrPrecision as number[][];
        precisionArr = precision(confusionMatrix);
    } else {
        // It's precision values array
        precisionArr = confusionMatrixOrPrecision as number[];
    }

    // Return the unweighted mean of all precision scores
    return macroAverage(precisionArr);
}

/**
 * Computes the weighted-averaged precision from precision values and confusion matrix.
 *
 * Weighted average calculates precision scores weighted by the number of predicted instances
 * for each class (column sums). This accounts for class imbalance in predictions.
 *
 * @param precision - The precision values for each class as an array.
 * @param confusionMatrix - The confusion matrix used to determine class weights (column sums = predicted instances).
 * @returns The weighted-averaged precision as a single number.
 */
export function weightedPrecision(precision: number[], confusionMatrix: number[][]): number;

/**
 * Computes the weighted-averaged precision from a confusion matrix.
 *
 * Weighted average calculates precision scores weighted by the number of predicted instances
 * for each class (column sums). This accounts for class imbalance in predictions.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The weighted-averaged precision as a single number.
 */
export function weightedPrecision(confusionMatrix: number[][]): number;

/**
 * Computes the weighted-averaged precision.
 *
 * The weighted average is calculated by:
 * 1. Computing precision for each class
 * 2. Weighting each precision score by the number of predicted instances (column sum) for that class
 * 3. Taking the weighted mean: sum(precision_i * predicted_i) / sum(predicted_i)
 *
 * This metric accounts for class imbalance in predictions by giving more weight to classes
 * with more predicted samples, making it useful when you want performance metrics that reflect
 * the actual distribution of predictions in your dataset.
 *
 * @param confusionMatrixOrPrecision - The confusion matrix or precision values.
 * @param confusionMatrixForWeights - The confusion matrix for weights (optional, only used when first param is precision array).
 * @returns The weighted-averaged precision as a single number.
 */
export function weightedPrecision(
    confusionMatrixOrPrecision: number[][] | number[],
    confusionMatrixForWeights?: number[][],
): number {
    let precisionArr: number[];
    let confusionMatrix: number[][];

    // Check if first parameter is a confusion matrix (2D array) or precision values (1D array)
    if (Array.isArray(confusionMatrixOrPrecision[0])) {
        // It's a confusion matrix - compute precision from it
        confusionMatrix = confusionMatrixOrPrecision as number[][];
        precisionArr = precision(confusionMatrix);
    } else {
        // It's precision values array
        precisionArr = confusionMatrixOrPrecision as number[];
        confusionMatrix = confusionMatrixForWeights!;
    }

    // Return the weighted mean of precision scores using column sums (predicted instances) as weights
    // Use false for useRowSums since precision is weighted by predicted positives (column sums)
    return weightedAverage(precisionArr, confusionMatrix, false);
}

/**
 * Computes the precision for a binary classification.
 *
 * @param binaryMatrix - The binary matrix as a 2D array ([[TP, FP], [FN, TN]]).
 * @returns The precision as a single number.
 */
export function binaryPrecision(binaryMatrix: number[][]): number {
    if (binaryMatrix.length !== 2 || binaryMatrix[0].length !== 2) {
        throw new Error('Binary matrix must be 2x2 for per-class metrics');
    }

    const [[TP], [FP]] = binaryMatrix;

    return TP + FP > 0 ? TP / (TP + FP) : 0;
}
