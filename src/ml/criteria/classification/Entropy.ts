import { EPSILON } from '../../constants';
import { BaseFunction } from '../BaseFunction';
import { CategoricalCrossentropy } from '../../losses';

export class Entropy extends BaseFunction {
    constructor(loss = new CategoricalCrossentropy()) {
        super(loss);
    }

    /**
     * Computes the entropy (information content) for a set of class labels.
     *
     * Formula:
     *     Entropy = -Σ(p_i * log(p_i))
     *
     * where p_i is the proportion of samples belonging to class i.
     *
     * @param yValues - The class labels or probabilities.
     * @returns Scalar representing the entropy.
     */
    impurity(yValues: number[][]): number {
        if (yValues.length === 0) {
            return 0;
        }

        const totalSamples = yValues.length;
        const numClasses = yValues[0]?.length || 0;

        if (numClasses === 0) {
            return 0;
        }

        // Calculate class counts by summing each column
        const classCounts = new Array(numClasses).fill(0);
        for (let i = 0; i < totalSamples; i++) {
            const row = yValues[i];
            for (let j = 0; j < numClasses; j++) {
                classCounts[j] += row[j];
            }
        }

        // Calculate class probabilities
        const classProbabilities = classCounts.map((count) => count / totalSamples);

        // Calculate entropy: -Σ(p_i * log(p_i))
        let entropy = 0;
        for (let i = 0; i < classProbabilities.length; i++) {
            const prob = classProbabilities[i];
            if (prob > EPSILON) {
                // Avoid log(0)
                entropy -= prob * Math.log(prob);
            }
        }

        return entropy;
    }
}
