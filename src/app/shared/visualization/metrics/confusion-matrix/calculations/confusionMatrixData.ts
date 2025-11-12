import type {
    BinaryConfusionMatrixMetrics,
    ConfusionMatrixData,
    MulticlassConfusionMatrixMetrics,
} from '../types';
import {
    accuracy as accuracyMetric,
    f1Scores as f1ScoresMetric,
    precision as precisionMetric,
    recall as recallMetric,
    mcc as mccMetric,
    cohensKappa as cohensKappaMetric,
    binaryPrecision,
    binaryRecall,
    binaryF1Score,
    macroAverage,
    weightedAverage,
} from '@/ml/metrics';

export function confusionMatrixData(
    confusionMatrix: number[][],
    numClasses: number,
): ConfusionMatrixData {
    const isBinaryClassification = numClasses === 2;

    if (isBinaryClassification) {
        return {
            matrix: confusionMatrix,
            metrics: binaryConfusionMatrixMetrics(confusionMatrix),
        };
    }

    const perClassMatrix: number[][][] = [];
    const perClassMetrics: BinaryConfusionMatrixMetrics[] = [];
    for (let i = 0; i < numClasses; i++) {
        const matrix = transformToOneVsRest(confusionMatrix, i);
        const metrics = binaryConfusionMatrixMetrics(matrix);
        perClassMatrix.push(matrix);
        perClassMetrics.push(metrics);
    }
    const metrics = multiclassConfusionMatrixMetrics(confusionMatrix);

    return {
        matrix: confusionMatrix,
        metrics,
        perClassMatrix,
        perClassMetrics,
    };
}

/**
 * Calculates the multiclass confusion matrix metrics
 *
 * @param confusionMatrix - The confusion matrix to calculate metrics for
 * @returns The multiclass confusion matrix metrics
 */
export function multiclassConfusionMatrixMetrics(
    confusionMatrix: number[][],
): MulticlassConfusionMatrixMetrics {
    const accuracy = accuracyMetric(confusionMatrix);
    const precisions = precisionMetric(confusionMatrix);
    const recalls = recallMetric(confusionMatrix);
    const f1Scores = f1ScoresMetric(precisions, recalls);
    const macroPrecision = macroAverage(precisions);
    const macroRecall = macroAverage(recalls);
    const macroF1 = macroAverage(f1Scores);
    const weightedPrecision = weightedAverage(precisions, confusionMatrix, false);
    const weightedRecall = weightedAverage(recalls, confusionMatrix, true);
    const weightedF1 = weightedAverage(f1Scores, confusionMatrix, true);
    const mcc = mccMetric(confusionMatrix);
    const cohensKappa = cohensKappaMetric(confusionMatrix);

    return {
        type: 'multiclass',
        accuracy,
        mcc,
        cohensKappa,
        macroPrecision,
        macroRecall,
        macroF1,
        weightedPrecision,
        weightedRecall,
        weightedF1,
    };
}

/**
 * Calculates the binary confusion matrix metrics
 *
 * @param confusionMatrix - The confusion matrix to calculate metrics for
 * @returns The binary confusion matrix metrics
 */
export function binaryConfusionMatrixMetrics(
    confusionMatrix: number[][],
): BinaryConfusionMatrixMetrics {
    const accuracy = accuracyMetric(confusionMatrix);
    const precision = binaryPrecision(confusionMatrix);
    const recall = binaryRecall(confusionMatrix);
    const f1 = binaryF1Score(precision, recall);
    const mcc = mccMetric(confusionMatrix);
    const cohensKappa = cohensKappaMetric(confusionMatrix);

    return {
        type: 'binary',
        accuracy,
        mcc,
        cohensKappa,
        precision,
        recall,
        f1,
    };
}

/**
 * Transforms a full confusion matrix into a one-vs-rest binary confusion matrix
 * for a specific class.
 *
 * Matrix format: matrix[row][col] where row = Expected (actual), col = Predicted
 *
 * @param matrix - The full confusion matrix (n x n), where matrix[row][col] represents
 *                  count of instances with Expected=row and Predicted=col
 * @param targetClassIndex - The index of the class to use for one-vs-rest transformation
 * @returns A 2x2 binary confusion matrix: [[TP, FN], [FP, TN]]
 *          Row 0 = Expected Positive (target class), Row 1 = Expected Negative (rest)
 *          Col 0 = Predicted Positive (target class), Col 1 = Predicted Negative (rest)
 */
export function transformToOneVsRest(matrix: number[][], targetClassIndex: number): number[][] {
    const size = matrix.length;

    if (targetClassIndex < 0 || targetClassIndex >= size) {
        throw new Error(
            `Invalid target class index: ${targetClassIndex}. Must be between 0 and ${size - 1}`,
        );
    }

    // True Positives: Expected = target class, Predicted = target class
    const truePositives = matrix[targetClassIndex][targetClassIndex];

    // Calculate row sum (target class row) and column sum (target class column)
    let rowSum = 0;
    let colSum = 0;
    let totalSum = 0;

    for (let i = 0; i < size; i++) {
        rowSum += matrix[targetClassIndex][i];
        colSum += matrix[i][targetClassIndex];

        // Calculate total sum while iterating
        for (let j = 0; j < size; j++) {
            totalSum += matrix[i][j];
        }
    }

    // False Negatives: Expected = target class, Predicted != target class
    // = sum of entire row minus TP
    const falseNegatives = rowSum - truePositives;

    // False Positives: Expected != target class, Predicted = target class
    // = sum of entire column minus TP
    const falsePositives = colSum - truePositives;

    // True Negatives: Expected != target class, Predicted != target class
    // = total sum minus TP, FN, and FP
    const trueNegatives = totalSum - truePositives - falseNegatives - falsePositives;

    // Return format: [[TP, FN], [FP, TN]]
    // Row 0 (Expected Positive): [TP, FN]
    // Row 1 (Expected Negative): [FP, TN]
    return [
        [truePositives, falseNegatives],
        [falsePositives, trueNegatives],
    ];
}
