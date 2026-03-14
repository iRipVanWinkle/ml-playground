import { tensor2d, tidy, type Tensor2D } from '@tensorflow/tfjs';
import type { TreeNode } from '../types';
import { Randomizer } from '../random/Randomizer';

/**
 * Creates a 2D array filled with zeros.
 * @param shape - Shape of the 2D array to create.
 * @returns A 2D array filled with zeros.
 */
export function zeros(shape: number[]): number[][] {
    return Array.from({ length: shape[0] }, () => Array(shape[1]).fill(0));
}

/**
 * Gathers rows from a 2D array based on provided indices.
 * @param features - The 2D array to gather rows from.
 * @param indices - The indices of the rows to gather.
 * @returns A 2D array containing the gathered rows.
 */
export function gather(features: number[][], indices: number[]): number[][] {
    if (indices.length === 0) {
        return zeros([0, features[0].length]);
    }

    return indices.map((idx) => features[idx]);
}

/**
 * Extracts values from a specific column in a 2D array based on provided row indices.
 * @param features - The 2D array to extract values from.
 * @param indexes - The indices of the rows to extract values from.
 * @param columnIndex - The index of the column to extract values from.
 * @returns An array containing the extracted values.
 */
export function getColumnValues(
    features: number[][],
    indexes: number[],
    columnIndex: number,
): number[] {
    return indexes.map((idx) => features[idx][columnIndex]);
}

/**
 * Splits indices into left and right groups based on a threshold applied to feature values.
 * @param featureValues - The feature values corresponding to the indices.
 * @param indices - The indices to be split.
 * @param threshold - The threshold value for splitting.
 * @returns An object containing leftIndices and rightIndices arrays.
 */
export function splitIndices(
    featureValues: number[],
    indices: number[],
    threshold: number,
): { leftIndices: number[]; rightIndices: number[] } {
    const leftIndices: number[] = [];
    const rightIndices: number[] = [];

    for (let i = 0; i < indices.length; i++) {
        const idx = indices[i];
        if (featureValues[i] <= threshold) {
            leftIndices.push(idx);
        } else {
            rightIndices.push(idx);
        }
    }

    return { leftIndices, rightIndices };
}

/**
 * Find the leaf node for a given sample in the decision tree.
 * @param sampleFeatures The features of the sample to classify.
 * @param rootNode The root node of the decision tree.
 * @returns The leaf node corresponding to the sample.
 */
export function findLeafNode(sampleFeatures: number[], rootNode: TreeNode): TreeNode {
    let node = rootNode;

    while (node.threshold !== null) {
        const shouldGoLeft = sampleFeatures[node.featureIndex!] < node.threshold;
        const nextNode = shouldGoLeft ? node.leftChild : node.rightChild;

        if (!nextNode) {
            break;
        }

        node = nextNode;
    }

    return node;
}

/**
 * Compute the mean value for regression tasks.
 * @param targetSamples The target samples to compute the mean from.
 * @returns The mean value.
 */
export function computeMeanValue(targetSamples: number[][]): { value: number } {
    // Calculate the mean of the targets
    return { value: targetSamples.reduce((sum, val) => sum + val[0], 0) / targetSamples.length };
}

/**
 * Compute the class probabilities for classification tasks.
 * @param targets The target samples to compute the probabilities from.
 * @returns The class probabilities.
 */
export function computeClassProbabilities(targets: number[][]): {
    value: number;
    probabilities: number[];
} {
    const classCounts = targets[0].map((_, colIndex) =>
        targets.reduce((sum, row) => sum + row[colIndex], 0),
    );

    const total = targets.length; // number of samples

    const classProbs = classCounts.map((count) => count / total); // shape: [numClasses]
    const predictedClass = classProbs.indexOf(Math.max(...classProbs));

    return {
        value: predictedClass,
        probabilities: classProbs,
    };
}

/**
 * Convert class probabilities to class indices.
 * @param probability The probability matrix.
 * @returns The class indices.
 */
export function probabilityToClassIndex(probability: Tensor2D): Tensor2D {
    return tidy(() => {
        // Find the indices of the maximum probabilities
        const maxIndices = probability.argMax(1);

        return maxIndices.reshape([-1, 1]);
    });
}

/**
 * Create a bootstrapped sample from the original dataset.
 * @param features The feature matrix.
 * @param targets The target vector.
 * @param seed The current iteration number (used for seeding).
 * @returns A bootstrapped sample of features and targets.
 */
export function bootstrapSample(
    features: Tensor2D,
    targets: Tensor2D,
    seed: number,
): [Tensor2D, Tensor2D] {
    const numSamples = features.shape[0];
    return tidy(() => {
        const indicesTensor = Randomizer.randomUniform([numSamples], 0, numSamples, 'int32', seed);

        // Gather the rows for features and targets using the sampled indices
        const bootstrappedFeatures = features.gather(indicesTensor);
        const bootstrappedTargets = targets.gather(indicesTensor);

        return [bootstrappedFeatures, bootstrappedTargets];
    });
}

/**
 * Create a subsampled feature matrix by randomly selecting rows without replacement.
 * @param features The feature matrix.
 * @param sampleSize Optional size of the subsampled dataset (defaults to the original dataset size).
 * @param seed The random seed for reproducibility.
 * @returns A subsampled dataset of features.
 */
export function subsampleFeatures(features: Tensor2D, sampleSize: number, seed: number): Tensor2D {
    const numSamples = features.shape[0];
    const actual = Math.min(sampleSize, numSamples);

    return tidy(() => {
        const indicesTensor = Randomizer.randomUniqueNumber(
            [actual],
            0,
            numSamples - 1,
            'int32',
            seed,
        );
        const subsampleFeatures = features.gather(indicesTensor);

        return subsampleFeatures;
    });
}

/**
 * Create a bootstrapped features from the original dataset.
 * @param features The feature matrix.
 * @param sampleSize The number of samples to draw.
 * @param seed The random seed for reproducibility.
 * @returns A bootstrapped dataset of features.
 */
export function bootstrapFeatures(features: Tensor2D, sampleSize: number, seed: number): Tensor2D {
    const numSamples = features.shape[0];
    const actual = Math.min(sampleSize, numSamples);

    return tidy(() => {
        const indicesTensor = Randomizer.randomUniform([actual], 0, numSamples, 'int32', seed);
        const subsampleFeatures = features.gather(indicesTensor);

        return subsampleFeatures;
    });
}

export function packTree(rootNode: TreeNode, numClasses = 0): Tensor2D {
    const nodes: TreeNode[] = [];
    const nodeIndexMap = new Map<TreeNode, number>();
    const queue: TreeNode[] = [rootNode];

    const adjustArrayLength = (arr: number[]) =>
        Array.from({ length: numClasses }, (_, i) => arr[i] ?? 0);

    // Breadth-first traversal to collect and index nodes
    while (queue.length > 0) {
        const node = queue.shift()!;
        nodeIndexMap.set(node, nodes.length);
        nodes.push(node);

        if (node.leftChild) queue.push(node.leftChild);
        if (node.rightChild) queue.push(node.rightChild);
    }

    // Flatten all node data into a single array
    const flatData: number[] = [numClasses, nodes.length]; // Start with the number of classes and nodes
    // Add each node's data: [featureIndex, threshold, leftChildIdx, rightChildIdx, value, probabilities]
    nodes.forEach((node) => {
        flatData.push(
            node.featureIndex ?? -1,
            node.threshold ? node.threshold : -1,
            node.leftChild ? nodeIndexMap.get(node.leftChild)! : -1,
            node.rightChild ? nodeIndexMap.get(node.rightChild)! : -1,
            node.value,
            ...adjustArrayLength(node.probabilities ?? []), // Add probabilities if they exist
        );
    });

    // Return as a column vector [n, 1]
    return tensor2d(flatData.map((val) => [val]));
}

export function unpackTree(flatTree: Tensor2D): TreeNode {
    const flatData = flatTree.arraySync() as number[][];
    const flatArray = flatData.flat(); // Extract values from [n, 1] shape
    const [numClasses, numNodes] = flatArray;
    const numFields = 5 + numClasses; // 5 fields per node + probabilities

    if (numNodes === 0) {
        throw new Error('Cannot unpack tree: no nodes found');
    }

    // Extract node data (6 values per node: featureIndex, threshold, value, leftChildIdx, rightChildIdx)
    const nodes: TreeNode[] = [];

    for (let i = 0; i < numNodes; i++) {
        const startIdx = 2 + i * numFields;
        const [
            featureIndex,
            threshold,
            ,
            ,
            /* leftChildIdx */ /* rightChildIdx */ value,
            ...probabilities
        ] = flatArray.slice(startIdx, startIdx + numFields);

        const node: TreeNode = {
            featureIndex: featureIndex === -1 ? null : featureIndex,
            threshold: threshold === -1 ? null : threshold,
            leftChild: null,
            rightChild: null,
            value,
            probabilities,
        };

        nodes.push(node);
    }

    // Link children by indices in a second pass
    for (let i = 0; i < numNodes; i++) {
        const startIdx = 2 + i * numFields;
        const [, , /* featureIndex */ /* threshold */ leftChildIdx, rightChildIdx] =
            flatArray.slice(startIdx, startIdx + numFields);

        if (leftChildIdx >= 0 && leftChildIdx < nodes.length) {
            nodes[i].leftChild = nodes[leftChildIdx];
        }
        if (rightChildIdx >= 0 && rightChildIdx < nodes.length) {
            nodes[i].rightChild = nodes[rightChildIdx];
        }
    }

    return nodes[0]; // Return root node
}
