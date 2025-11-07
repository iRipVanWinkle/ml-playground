import { macroAverage, weightedAverage } from './utils';

/**
 * Computes the recall metric for each class from a confusion matrix.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The recall for each class as an array.
 */
export function recall(confusionMatrix: number[][]): number[] {
    const numClasses = confusionMatrix.length;
    const recalls: number[] = [];

    // i = row = expected/actual class
    for (let i = 0; i < numClasses; i++) {
        let actualPositives = 0;
        let truePositives = 0;

        // j = col = predicted class
        for (let j = 0; j < numClasses; j++) {
            actualPositives += confusionMatrix[i][j];
            if (i === j) {
                truePositives = confusionMatrix[i][j];
            }
        }

        // Recall = TP / all actual positives
        recalls[i] = actualPositives > 0 ? truePositives / actualPositives : 0;
    }

    return recalls;
}

/**
 * Computes the macro-averaged recall from recall values.
 *
 * Macro average calculates the unweighted mean of recall scores across all classes,
 * treating each class equally regardless of its frequency in the dataset.
 *
 * @param recall - The recall values for each class as an array.
 * @returns The macro-averaged recall as a single number.
 */
export function macroRecall(recall: number[]): number;

/**
 * Computes the macro-averaged recall from a confusion matrix.
 *
 * Macro average calculates the unweighted mean of recall scores across all classes,
 * treating each class equally regardless of its frequency in the dataset.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The macro-averaged recall as a single number.
 */
export function macroRecall(confusionMatrix: number[][]): number;

/**
 * Computes the macro-averaged recall.
 *
 * The macro average is calculated by:
 * 1. Computing recall for each class
 * 2. Taking the unweighted mean of all recall scores
 *
 * This metric treats all classes equally, making it useful when you want to
 * evaluate performance across all classes without bias toward more frequent classes.
 *
 * @param confusionMatrixOrRecall - The confusion matrix or recall values.
 * @returns The macro-averaged recall as a single number.
 */
export function macroRecall(confusionMatrixOrRecall: number[][] | number[]): number {
    let recallArr: number[];

    // Check if first parameter is a confusion matrix (2D array) or recall values (1D array)
    if (Array.isArray(confusionMatrixOrRecall[0])) {
        // It's a confusion matrix - compute recall from it
        const confusionMatrix = confusionMatrixOrRecall as number[][];
        recallArr = recall(confusionMatrix);
    } else {
        // It's recall values array
        recallArr = confusionMatrixOrRecall as number[];
    }

    // Return the unweighted mean of all recall scores
    return macroAverage(recallArr);
}

/**
 * Computes the weighted-averaged recall from recall values and confusion matrix.
 *
 * Weighted average calculates recall scores weighted by the number of true instances
 * for each class (row sums). This accounts for class imbalance in the dataset.
 *
 * @param recall - The recall values for each class as an array.
 * @param confusionMatrix - The confusion matrix used to determine class weights (row sums = support).
 * @returns The weighted-averaged recall as a single number.
 */
export function weightedRecall(recall: number[], confusionMatrix: number[][]): number;

/**
 * Computes the weighted-averaged recall from a confusion matrix.
 *
 * Weighted average calculates recall scores weighted by the number of true instances
 * for each class (row sums). This accounts for class imbalance in the dataset.
 *
 * @param confusionMatrix - The confusion matrix as a 2D array (row - expected/actual class, col - predicted class).
 * @returns The weighted-averaged recall as a single number.
 */
export function weightedRecall(confusionMatrix: number[][]): number;

/**
 * Computes the weighted-averaged recall.
 *
 * The weighted average is calculated by:
 * 1. Computing recall for each class
 * 2. Weighting each recall score by the number of true instances (support) for that class
 * 3. Taking the weighted mean: sum(recall_i * support_i) / sum(support_i)
 *
 * This metric accounts for class imbalance by giving more weight to classes with
 * more samples, making it useful when you want performance metrics that reflect
 * the actual distribution of classes in your dataset.
 *
 * @param confusionMatrixOrRecall - The confusion matrix or recall values.
 * @param confusionMatrixForWeights - The confusion matrix for weights (optional, only used when first param is recall array).
 * @returns The weighted-averaged recall as a single number.
 */
export function weightedRecall(
    confusionMatrixOrRecall: number[][] | number[],
    confusionMatrixForWeights?: number[][],
): number {
    let recallArr: number[];
    let confusionMatrix: number[][];

    // Check if first parameter is a confusion matrix (2D array) or recall values (1D array)
    if (Array.isArray(confusionMatrixOrRecall[0])) {
        // It's a confusion matrix - compute recall from it
        confusionMatrix = confusionMatrixOrRecall as number[][];
        recallArr = recall(confusionMatrix);
    } else {
        // It's recall values array
        recallArr = confusionMatrixOrRecall as number[];
        confusionMatrix = confusionMatrixForWeights!;
    }

    // Return the weighted mean of recall scores using row sums (support/actual instances) as weights
    // Use true for useRowSums since recall is weighted by actual positives (row sums)
    return weightedAverage(recallArr, confusionMatrix, true);
}

/**
 * Computes the recall for a binary classification.
 *
 * @param binaryMatrix - The binary matrix as a 2D array ([[TP, FN], [FP, TN]]).
 * @returns The recall as a single number.
 */
export function binaryRecall(binaryMatrix: number[][]): number {
    if (binaryMatrix.length !== 2 || binaryMatrix[0].length !== 2) {
        throw new Error('Binary matrix must be 2x2 for per-class metrics');
    }

    const [[TP, FN]] = binaryMatrix;

    return TP + FN > 0 ? TP / (TP + FN) : 0;
}
