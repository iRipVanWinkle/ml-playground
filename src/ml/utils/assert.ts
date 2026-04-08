/**
 * Asserts that a condition is true. If not, throws an error with the provided message.
 * @param condition - The condition to check.
 * @param message - The error message to throw if the assertion fails.
 */
export function assert(condition: boolean, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

/**
 * Asserts that the model has been trained by checking if the theta parameter is not null.
 * @param theta - The model parameters to check.
 */
export function assertModelTrained(value: unknown): asserts value is NonNullable<unknown> {
    const isNillable = value === null || value === undefined;
    const isEmptyArray = Array.isArray(value) && value.length === 0;

    assert(
        !isNillable && !isEmptyArray,
        'Model has not been trained yet. Please call train() first.',
    );
}
