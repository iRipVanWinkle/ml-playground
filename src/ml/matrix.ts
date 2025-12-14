import type { Tensor2D } from '@tensorflow/tfjs';

export type TypedArray = Float32Array | Uint8Array | Int32Array;

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
 * Gets a matrix from an array.
 * @param matrix - The array to get the matrix from.
 * @returns A matrix.
 */
export function getMatrixFromArray(matrix: number[][]): MatrixLike {
    const numRows = matrix.length;
    const numCols = matrix[0]?.length || 0;
    const array = new Float32Array(numRows * numCols);
    for (let i = 0; i < numRows; i++) {
        const rowOffset = i * numCols;
        for (let j = 0; j < numCols; j++) {
            array[rowOffset + j] = matrix[i][j];
        }
    }
    return { array, shape: [numRows, numCols] };
}

/**
 * A class representing a matrix.
 */
export class Matrix implements MatrixLike {
    array: TypedArray;
    shape: [number, number];

    constructor(matrix: MatrixLike) {
        this.array = matrix.array;
        this.shape = matrix.shape;
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
     * Creates a new matrix with the given shape and type.
     * @param shape - The shape of the matrix.
     * @param TypedArray - The type of the matrix.
     * @returns A new matrix.
     */
    static create(
        shape: [number, number],
        TypedArray: new (length: number) => TypedArray = Float32Array,
    ): Matrix {
        return new Matrix({
            array: new TypedArray(shape[0] * shape[1]),
            shape,
        });
    }

    /**
     * Gets the value of the matrix at the given index.
     * @param i - The index of the row.
     * @param j - The index of the column.
     * @returns The value of the matrix at the given index.
     */
    get(i: number, j: number): number {
        return this.array[i * this.shape[1] + j];
    }

    /**
     * Flattens the matrix into a 1D array.
     * @returns A 1D array.
     */
    flatten(): TypedArray {
        return this.array.slice();
    }

    /**
     * Gets a row of the matrix.
     * @param index - The index of the row.
     * @returns A 1D array.
     */
    row(index: number): TypedArray {
        return this.array.subarray(index * this.shape[1], (index + 1) * this.shape[1]);
    }

    /**
     * Gets a column of the matrix.
     * @param index - The index of the column.
     * @returns A 1D array.
     */
    col(index: number): TypedArray {
        const result = this.createTypedArray(this.shape[0]);
        for (let i = 0; i < this.shape[0]; i++) {
            result[i] = this.array[i * this.shape[1] + index];
        }
        return result;
    }

    /**
     * Gets the shape of the matrix.
     * @returns The shape of the matrix.
     */
    getShape(): [number, number] {
        return this.shape;
    }

    /**
     * Gets the array of the matrix.
     * @returns The array of the matrix.
     */
    getFlatArray(): TypedArray {
        return this.array;
    }

    private createTypedArray(length: number): TypedArray {
        const TypedArrayConstructor = this.array.constructor as new (length: number) => TypedArray;
        return new TypedArrayConstructor(length);
    }
}
