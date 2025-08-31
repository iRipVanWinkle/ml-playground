import { Tensor, tidy, type Tensor2D } from '@tensorflow/tfjs';

/**
 * Soft Voting: Average the predicted probabilities and return the class with the highest average probability
 * @param probs Tensor3D of shape [num_samples, num_trees, num_classes]
 * @returns Tensor2D of shape [num_samples, num_classes] (soft voting probabilities)
 */
export function softVoting(probs: Tensor): Tensor2D {
    if (probs.shape.length !== 3) {
        throw new Error('Input tensor must be 3D');
    }

    return tidy(() => probs.mean(1));
}
