import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { CriterionFunction } from '../../types';

export class Gini implements CriterionFunction {
    /**
     * Computes the Gini impurity for a set of class labels.
     *
     * Formula:
     *     Gini = 1 - Σ(p_i)²
     *
     * where p_i is the proportion of samples belonging to class i.
     *
     * @param yValues - The class labels or probabilities.
     * @returns Scalar representing the Gini impurity.
     */
    impurity(yValues: number[][]): number {
        // Calculate class probabilities
        const numSamples = yValues.length;
        const numClasses = yValues[0]?.length || 0;

        // Sum along samples axis to get class counts
        const classCounts = new Array(numClasses).fill(0);
        for (let i = 0; i < numSamples; i++) {
            const row = yValues[i];
            for (let j = 0; j < numClasses; j++) {
                classCounts[j] += row[j];
            }
        }

        // Gini impurity: 1 - Σ(p_i)²
        let sumSquaredProbabilities = 0;
        for (let i = 0; i < classCounts.length; i++) {
            const prob = classCounts[i] / numSamples;
            sumSquaredProbabilities += prob * prob;
        }
        const giniImpurityValue = 1 - sumSquaredProbabilities;

        return giniImpurityValue;
    }

    /**
     * Gini impurity is a measure of how often a randomly chosen element from the set would be
     * incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
     *
     * Formula for multiclass:
     *     Gini = 1 - Σ(p_i)²
     *
     * where:
     *     - p_i: probability of class i
     *     - The sum is over all classes
     *
     * For binary classification (special case):
     *     Gini = 2 * p * (1 - p)
     *
     * @param yTrue - The true values (one-hot encoded for multiclass).
     * @param yPred - The predicted probabilities.
     * @returns Scalar representing the Gini impurity.
     */
    loss(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return tidy(() => {
            // For classification loss, we use cross-entropy-like formulation with Gini coefficient
            // This adapts Gini for use as a loss function in gradient-based optimization
            const loss = yTrue.mul(yPred.square().neg().add(1));

            return loss.mean();
        });
    }
}
