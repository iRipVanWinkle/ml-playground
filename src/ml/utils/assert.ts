import type { Tensor2D } from '@tensorflow/tfjs';

/**
 * Asserts that a condition is true. If not, throws an error with the provided message.
 * @param condition - The condition to check.
 * @param message - The error message to throw if the assertion fails.
 */
export function assert(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
}

/**
 * Asserts that the model has been trained by checking if the theta parameter is not null.
 * @param theta - The model parameters to check.
 */
export function assertThetaTrained(theta: Tensor2D | null): void {
    assert(!!theta, 'Model has not been trained yet. Please call train() first.');
}
