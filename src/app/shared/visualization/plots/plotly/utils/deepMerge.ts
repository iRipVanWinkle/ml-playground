type DeepPartial<T> = T extends object ? { [P in keyof T]?: DeepPartial<T[P]> } : T;

/**
 * Deep merge two objects
 * @param target - The target object
 * @param source - The source object
 * @returns The merged object with the same type as target
 */
export function deepMerge<T extends Record<string, unknown>>(
    target: T,
    source: DeepPartial<NoInfer<T>>,
): T {
    const result = { ...target } as Record<string, unknown>;

    for (const key of Object.keys(source)) {
        const targetValue = target[key as keyof typeof target];
        const sourceValue = source[key as keyof typeof source];

        if (isPlainObject(targetValue) && isPlainObject(sourceValue)) {
            result[key] = deepMerge(
                targetValue as Record<string, unknown>,
                sourceValue as Record<string, unknown>,
            );
        } else {
            result[key] = sourceValue;
        }
    }

    return result as T;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
