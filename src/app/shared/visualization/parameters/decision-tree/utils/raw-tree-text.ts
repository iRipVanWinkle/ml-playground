import type { TreeNode } from '@/ml/types';

export const generateRawModelText = (
    node: TreeNode,
    featureLabels: string[],
    categories?: string[],
    depth = 0,
): string => {
    const indent = '|--- '.repeat(depth);
    const isLeaf = !node.leftChild && !node.rightChild;

    if (isLeaf) {
        if (node.probabilities) {
            // Classification
            const maxProb = Math.max(...node.probabilities);
            const classIndex = node.probabilities.indexOf(maxProb);
            return `${indent}class: ${categories?.[classIndex] ?? classIndex} (prob: ${maxProb.toFixed(4)})\n`;
        }
        // Regression
        return `${indent}value: ${node.value.toFixed(4)}\n`;
    }

    // Internal node
    const featureName =
        node.featureIndex !== null && featureLabels[node.featureIndex]
            ? featureLabels[node.featureIndex]
            : `feature_${node.featureIndex}`;

    return (
        `${indent}${featureName} <= ${node.threshold?.toFixed(4)}\n` +
        (node.leftChild
            ? generateRawModelText(node.leftChild, featureLabels, categories, depth + 1)
            : '') +
        (node.rightChild
            ? generateRawModelText(node.rightChild, featureLabels, categories, depth + 1)
            : '')
    );
};
