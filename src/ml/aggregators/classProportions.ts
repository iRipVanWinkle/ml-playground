import { oneHot, ones, tidy, type Tensor2D } from '@tensorflow/tfjs';

export function classProportions(labels: Tensor2D, numClasses: number): Tensor2D {
    if (labels.shape.length !== 2) {
        throw new Error('Labels tensor must be 2D');
    }

    if (!Number.isInteger(numClasses) || numClasses < 1) {
        throw new Error('numClasses must be a positive integer');
    }

    return tidy(() => {
        const [numSamples, numVotes] = labels.shape;

        const labelsFlat = labels.reshape([-1]).cast('int32');
        const hot = oneHot(labelsFlat, numClasses);

        const reshaped = hot.reshape([numSamples, numVotes, numClasses]);

        // THE WEBGPU FIX (Mimicking weightedClassProportions):
        // 1. Create a float32 tensor of 1.0s.
        const dummyWeights = ones([numSamples, numVotes, 1], 'float32');

        // 2. Multiply the int32 one-hot matrix by the float32 dummy weights.
        // This forces a safe, implicit float promotion via the stable .mul() shader.
        const floatVotes = reshaped.mul(dummyWeights);

        // 3. Sum across Axis 1 and divide to get the mean
        const sums = floatVotes.sum(1);
        return sums.div(numVotes) as Tensor2D;
    });
}
