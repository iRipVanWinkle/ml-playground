import type { MatrixLike, TypedArray } from '@/app/shared/helpers';

type ClassParameters = {
    classIndex: number;
    weights: TypedArray;
    bias: number;
    min: number;
    max: number;
};

export function useClassParameters(theta: MatrixLike) {
    const [numClasses, numFeatures] = theta.shape;

    const params: Array<ClassParameters> = [];
    const numWeights = numFeatures - 1; // exclude bias

    for (let classIndex = 0; classIndex < numClasses; classIndex++) {
        const rowStart = classIndex * numFeatures;
        const bias = theta.array[rowStart];

        let min = Infinity;
        let max = -Infinity;

        const weights = theta.array.subarray(rowStart + 1, rowStart + numFeatures);
        for (let i = 0; i < numWeights; i++) {
            const weight = weights[i];

            if (weight < min) min = weight;
            if (weight > max) max = weight;
        }

        params.push({ classIndex, weights, bias, min, max });
    }
    return params;
}
