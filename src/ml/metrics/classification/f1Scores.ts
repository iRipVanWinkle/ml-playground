import { binaryPrecision, precision } from './precision';
import { binaryRecall, recall } from './recall';
import { macroAverage, weightedAverage } from './utils';

/**
 * Computes the F1 score metric from precision and recall values.
 *
 * @param precision - The precision values for each class as an array.
 * @param recall - The recall values for each class as an array.
 * @returns The F1 score for each class as an array.
 */
export function f1Scores(precision: number[], recall: number[]): number[];

/**
 * Computes the F1 score metric for each class from a confusion matrix.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The F1 score for each class as an array.
 */
export function f1Scores(confusionMatrix: number[][]): number[];

/**
 * Computes the F1 score metric.
 * F1 = 2 * (precision * recall) / (precision + recall)
 *
 * @param confusionMatrixOrPrecision - The confusion matrix or precision values.
 * @param recall - The recall values (optional, only used when first param is precision array).
 * @returns The F1 score for each class as an array.
 */
export function f1Scores(
    confusionMatrixOrPrecision: number[][] | number[],
    recallValues?: number[],
): number[] {
    let precisionArr: number[];
    let recallArr: number[];

    // Check if first parameter is a confusion matrix (2D array) or precision values (1D array)
    if (Array.isArray(confusionMatrixOrPrecision[0])) {
        // It's a confusion matrix
        const confusionMatrix = confusionMatrixOrPrecision as number[][];
        precisionArr = precision(confusionMatrix);
        recallArr = recall(confusionMatrix);
    } else {
        // It's precision values array
        precisionArr = confusionMatrixOrPrecision as number[];
        recallArr = recallValues!;
    }

    // Compute F1 score for each class
    const f1Scores: number[] = [];
    for (let i = 0; i < precisionArr.length; i++) {
        const p = precisionArr[i];
        const r = recallArr[i];

        // F1 = 2 * (precision * recall) / (precision + recall)
        f1Scores[i] = p + r > 0 ? (2 * p * r) / (p + r) : 0;
    }

    return f1Scores;
}

/**
 * Computes the macro-averaged F1 score from precision and recall values.
 *
 * Macro average calculates the unweighted mean of F1 scores across all classes,
 * treating each class equally regardless of its frequency in the dataset.
 *
 * @param precision - The precision values for each class as an array.
 * @param recall - The recall values for each class as an array.
 * @returns The macro-averaged F1 score as a single number.
 */
export function macroAverageF1Score(precision: number[], recall: number[]): number;

/**
 * Computes the macro-averaged F1 score from a confusion matrix.
 *
 * Macro average calculates the unweighted mean of F1 scores across all classes,
 * treating each class equally regardless of its frequency in the dataset.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The macro-averaged F1 score as a single number.
 */
export function macroAverageF1Score(confusionMatrix: number[][]): number;

/**
 * Computes the macro-averaged F1 score.
 *
 * The macro average is calculated by:
 * 1. Computing F1 score for each class
 * 2. Taking the unweighted mean of all F1 scores
 *
 * This metric treats all classes equally, making it useful when you want to
 * evaluate performance across all classes without bias toward more frequent classes.
 *
 * @param confusionMatrixOrPrecision - The confusion matrix or precision values.
 * @param recallValues - The recall values (optional, only used when first param is precision array).
 * @returns The macro-averaged F1 score as a single number.
 */
export function macroAverageF1Score(
    confusionMatrixOrPrecision: number[][] | number[],
    recallValues?: number[],
): number {
    let f1s: number[];

    // Check if first parameter is a confusion matrix (2D array) or precision values (1D array)
    if (Array.isArray(confusionMatrixOrPrecision[0])) {
        // It's a confusion matrix - compute F1 scores from it
        const confusionMatrix = confusionMatrixOrPrecision as number[][];
        f1s = f1Scores(confusionMatrix);
    } else {
        // It's precision values array - compute F1 scores from precision and recall
        const precisionArr = confusionMatrixOrPrecision as number[];
        f1s = f1Scores(precisionArr, recallValues!);
    }

    // Return the unweighted mean of all F1 scores
    return macroAverage(f1s);
}

/**
 * Computes the weighted-averaged F1 score from precision, recall values, and confusion matrix.
 *
 * Weighted average calculates F1 scores weighted by the number of true instances
 * for each class (support). This accounts for class imbalance in the dataset.
 *
 * @param precision - The precision values for each class as an array.
 * @param recall - The recall values for each class as an array.
 * @param confusionMatrix - The confusion matrix used to determine class weights (row sums = support).
 * @returns The weighted-averaged F1 score as a single number.
 */
export function weightedAverageF1Score(
    precision: number[],
    recall: number[],
    confusionMatrix: number[][],
): number;

/**
 * Computes the weighted-averaged F1 score from a confusion matrix.
 *
 * Weighted average calculates F1 scores weighted by the number of true instances
 * for each class (support). This accounts for class imbalance in the dataset.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The weighted-averaged F1 score as a single number.
 */
export function weightedAverageF1Score(confusionMatrix: number[][]): number;

/**
 * Computes the weighted-averaged F1 score.
 *
 * The weighted average is calculated by:
 * 1. Computing F1 score for each class
 * 2. Weighting each F1 score by the number of true instances (support) for that class
 * 3. Taking the weighted mean: sum(F1_i * support_i) / sum(support_i)
 *
 * This metric accounts for class imbalance by giving more weight to classes with
 * more samples, making it useful when you want performance metrics that reflect
 * the actual distribution of classes in your dataset.
 *
 * @param confusionMatrixOrPrecision - The confusion matrix or precision values.
 * @param recallValues - The recall values (optional, only used when first param is precision array).
 * @param confusionMatrixForWeights - The confusion matrix for weights (optional, only used when first param is precision array).
 * @returns The weighted-averaged F1 score as a single number.
 */
export function weightedAverageF1Score(
    confusionMatrixOrPrecision: number[][] | number[],
    recallValues?: number[],
    confusionMatrixForWeights?: number[][],
): number {
    let f1s: number[];
    let confusionMatrix: number[][];

    // Check if first parameter is a confusion matrix (2D array) or precision values (1D array)
    if (Array.isArray(confusionMatrixOrPrecision[0])) {
        // It's a confusion matrix - compute F1 scores from it
        confusionMatrix = confusionMatrixOrPrecision as number[][];
        f1s = f1Scores(confusionMatrix);
    } else {
        // It's precision values array - compute F1 scores from precision and recall
        const precisionArr = confusionMatrixOrPrecision as number[];
        f1s = f1Scores(precisionArr, recallValues!);
        confusionMatrix = confusionMatrixForWeights!;
    }

    // Return the weighted mean of F1 scores using row sums (support) as weights
    return weightedAverage(f1s, confusionMatrix, true);
}

/**
 * Computes the F1 score for a binary classification from precision and recall values.
 *
 * @param precision - The precision value as a single number.
 * @param recall - The recall value as a single number.
 * @returns The F1 score as a single number.
 */
export function binaryF1Score(precision: number, recall: number): number;

/**
 * Computes the F1 score for a binary classification from a binary confusion matrix.
 *
 * @param binaryMatrix - The binary matrix as a 2D array ([[TP, FN], [FP, TN]]).
 * @returns The F1 score as a single number.
 */
export function binaryF1Score(binaryMatrix: number[][]): number;

/**
 * Computes the F1 score for a binary classification.
 * F1 = 2 * (precision * recall) / (precision + recall)
 *
 * @param binaryMatrixOrPrecision - The binary matrix or precision value.
 * @param recallValue - The recall value (optional, only used when first param is precision).
 * @returns The F1 score as a single number.
 */
export function binaryF1Score(
    binaryMatrixOrPrecision: number[][] | number,
    recall?: number,
): number {
    let precisionValue: number;
    let recallValue: number;

    // Check if first parameter is a binary matrix (2D array) or precision (number)
    if (Array.isArray(binaryMatrixOrPrecision)) {
        // It's a binary matrix
        const binaryMatrix = binaryMatrixOrPrecision;
        if (binaryMatrix.length !== 2 || binaryMatrix[0].length !== 2) {
            throw new Error('Binary matrix must be 2x2 for per-class metrics');
        }

        // Calculate precision: TP / (TP + FP)
        precisionValue = binaryPrecision(binaryMatrix);

        // Calculate recall: TP / (TP + FN)
        recallValue = binaryRecall(binaryMatrix);
    } else {
        // It's precision value
        precisionValue = binaryMatrixOrPrecision;
        recallValue = recall!;
    }

    // F1 = 2 * (precision * recall) / (precision + recall)
    return precisionValue + recallValue > 0
        ? (2 * precisionValue * recallValue) / (precisionValue + recallValue)
        : 0;
}
