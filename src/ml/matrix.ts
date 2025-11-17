import type { Tensor2D } from '@tensorflow/tfjs';

type TypedArray = Float32Array | Uint8Array | Int32Array;

/**
 * A type representing a matrix-like object.
 */
export type MatrixLike = {
    array: TypedArray;
    shape: [number, number];
};

export const EMPTY_MATRIX_LIKE: MatrixLike = { array: new Uint8Array([]), shape: [0, 0] };

/**
 * Gets a matrix from a tensor.
 * @param tensor - The tensor to get the matrix from.
 * @returns A matrix.
 */
export async function getMatrixFromTensor(tensor: Tensor2D): Promise<MatrixLike> {
    const shape = tensor.shape as [number, number];
    const array = await tensor.data();
    return { array, shape };
}

/**
 * A class representing a matrix.
 */
export class Matrix {
    private matrix: MatrixLike;

    constructor(matrix: MatrixLike) {
        this.matrix = matrix;
    }

    /**
     * Creates a new matrix from a matrix-like object.
     * @param matrix - The matrix-like object.
     * @returns A new matrix.
     */
    static from(matrix: MatrixLike): Matrix {
        return new Matrix(matrix);
    }

    /**
     * Gets the value of the matrix at the given index.
     * @param i - The index of the row.
     * @param j - The index of the column.
     * @returns The value of the matrix at the given index.
     */
    get(i: number, j: number): number {
        return this.matrix.array[i * this.matrix.shape[1] + j];
    }

    /**
     * Flattens the matrix into a 1D array.
     * @returns A 1D array.
     */
    flatten(): TypedArray {
        return this.matrix.array.slice();
    }

    /**
     * Gets a row of the matrix.
     * @param index - The index of the row.
     * @returns A 1D array.
     */
    row(index: number): TypedArray {
        const result = this.createTypedArray(this.matrix.shape[1]);
        const offset = index * this.matrix.shape[1];
        for (let i = 0; i < this.matrix.shape[1]; i++) {
            result[i] = this.matrix.array[offset + i];
        }
        return result;
    }

    /**
     * Gets a column of the matrix.
     * @param index - The index of the column.
     * @returns A 1D array.
     */
    col(index: number): TypedArray {
        const result = this.createTypedArray(this.matrix.shape[0]);
        for (let i = 0; i < this.matrix.shape[0]; i++) {
            result[i] = this.matrix.array[i * this.matrix.shape[1] + index];
        }
        return result;
    }

    /**
     * Gets the shape of the matrix.
     * @returns The shape of the matrix.
     */
    getShape(): [number, number] {
        return this.matrix.shape;
    }

    /**
     * Gets the array of the matrix.
     * @returns The array of the matrix.
     */
    getFlatArray(): TypedArray {
        return this.matrix.array;
    }

    private createTypedArray(length: number): TypedArray {
        const TypedArrayConstructor = this.matrix.array.constructor as new (
            length: number,
        ) => TypedArray;
        return new TypedArrayConstructor(length);
    }
}
