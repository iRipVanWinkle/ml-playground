import { oneHot, tidy, type Tensor, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Hard voting for ensemble tree outputs.
 * @param probs Tensor3D of shape [num_samples, num_trees, num_classes]
 * @returns Tensor2D of shape [num_samples, num_classes] (hard voting probabilities)
 */
export function hardVoting(probs: Tensor): Tensor2D {
    if (probs.shape.length !== 3) {
        throw new Error('Input tensor must be 3D');
    }

    return tidy(() => {
        // For each tree, pick the class with highest probability
        const predictedClass = probs.argMax(2); // shape: [num_samples, num_trees]

        // One-hot encode the selected class per tree
        const predictedClassOneHot = oneHot(predictedClass, probs.shape[2]!); // shape: [num_samples, num_trees, num_classes]

        // Sum votes for each class across trees
        const votes = predictedClassOneHot.sum(1); // shape: [num_samples, num_classes]

        // Convert counts to probabilities
        return votes.div(probs.shape[1]!);
    });
}
