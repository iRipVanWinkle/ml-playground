/**
 * Represents either a 2D matrix or a 1D vector of numbers.
 */
type NumericArray = number[] | number[][];

/**
 * Creates an array of numbers from 0 to n-1.
 * @param n - The number of elements in the array.
 * @returns An array of numbers from 0 to n-1.
 */
export function range(n: number): number[] {
    return Array.from({ length: n }, (_, i) => i);
}

/**
 * Creates a 1D array filled with zeros.
 * @param shape - Shape of the 1D array to create.
 */
export function zeros(shape: [number]): number[];

/**
 * Creates a 2D array filled with zeros.
 * @param shape - Shape of the 2D array to create.
 */
export function zeros(shape: [number, number]): number[][];

export function zeros(shape: number[]): NumericArray {
    if (shape.length === 2) {
        return Array.from({ length: shape[0] }, () => Array(shape[1]).fill(0));
    }

    return Array.from({ length: shape[0] }, () => 0);
}

/**
 * Gathers rows from a 2D array based on provided indices.
 * @param features - The 2D array to gather rows from.
 * @param indices - The indices of the rows to gather.
 * @returns A 2D array containing the gathered rows.
 */
export function gather(features: number[][], indices: number[]): number[][];

/**
 * Gathers elements from a 1D array based on provided indices.
 * @param features - The 1D array to gather elements from.
 * @param indices - The indices of the elements to gather.
 * @returns A 1D array containing the gathered elements.
 */
export function gather(features: number[], indices: number[]): number[];

export function gather(features: NumericArray, indices: number[]): NumericArray {
    if (indices.length === 0) {
        return [];
    }

    return indices.map((idx) => features[idx]) as NumericArray;
}
