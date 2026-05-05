import { getBackend, type Tensor2D } from '@tensorflow/tfjs';
import { EPSILON } from '../constants';

export type TypedArray = Float32Array | Uint8Array | Int32Array;

/**
 * A type representing a matrix-like object.
 */
export type MatrixLike = {
    array: TypedArray;
    shape: [number, number];
};

/**
 * Returns a fresh, uniquely-owned empty matrix.
 */
export const createEmptyMatrix = (): MatrixLike => ({
    array: new Uint8Array(0),
    shape: [0, 0],
});

/**
 * Gets a matrix from a tensor.
 *
 * @param tensor - The tensor to get the matrix from.
 * @returns A matrix.
 */
export async function getMatrixFromTensor(tensor: Tensor2D): Promise<MatrixLike> {
    const shape = tensor.shape as [number, number];
    const data = await tensor.data();

    // On the `cpu` backend `tensor.data()` returns the tensor's own TypedArray, so
    // the result is sliced to give the caller a uniquely-owned buffer (safe to
    // transfer or hold past the tensor's lifetime). Other backends (`webgpu`,
    // `webgl`, `wasm`) already materialize a fresh allocation as part of the
    // GPU -> CPU / heap -> JS readback, so no extra copy is needed.
    const array = getBackend() === 'cpu' ? data.slice() : data;

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

/**
 * Computes the inverse and determinant of a positive-definite symmetric matrix.
 * Uses Cholesky decomposition.
 *
 * @param matrix - Square symmetric positive-definite matrix
 * @param epsilon - Small value to use for numerical stability on the diagonal
 * @returns Object with inverse matrix and determinant
 */
export function calculateInverseAndDeterminant(
    matrix: MatrixLike,
    epsilon: number = EPSILON,
): { inverse: Matrix; determinant: number } {
    const n = matrix.shape[0];
    const array = matrix.array;
    const L = new Float32Array(n * n);

    // Cholesky decomposition: A = LLᵀ
    for (let i = 0; i < n; i++) {
        const rowI = i * n;
        for (let j = 0; j <= i; j++) {
            const rowJ = j * n;
            let sum = 0;
            for (let k = 0; k < j; k++) {
                sum += L[rowI + k] * L[rowJ + k];
            }

            if (i === j) {
                L[rowI + j] = Math.sqrt(Math.max(array[rowI + i] - sum, epsilon));
            } else {
                L[rowI + j] = (array[rowI + j] - sum) / L[rowJ + j];
            }
        }
    }

    // Determinant is product of diagonal elements of L squared
    let determinant = 1;
    for (let i = 0; i < n; i++) {
        determinant *= L[i * n + i];
    }
    determinant = determinant * determinant;

    // Compute inverse using forward and backward substitution
    const inverse = Matrix.create([n, n]);
    const invArray = inverse.array;

    const y = new Float32Array(n);
    const x = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        // Forward substitution: solve Ly = b
        // b is a one-hot vector where b[i] = 1, otherwise 0
        // When j < i, b[j] = 0.
        // sum = sum(L[j, k] * y[k])
        // Since y[k] = 0 for k < i, we can skip calculating y[j] for j < i
        for (let j = 0; j < i; j++) {
            y[j] = 0;
        }

        for (let j = i; j < n; j++) {
            const rowJ = j * n;
            let sum = 0;
            // Since y[k] = 0 for k < i, we can start k from i
            for (let k = i; k < j; k++) {
                sum += L[rowJ + k] * y[k];
            }
            const bj = j === i ? 1 : 0;
            y[j] = (bj - sum) / L[rowJ + j];
        }

        // Backward substitution: solve Lᵀx = y
        for (let j = n - 1; j >= 0; j--) {
            let sum = 0;
            for (let k = j + 1; k < n; k++) {
                sum += L[k * n + j] * x[k];
            }
            x[j] = (y[j] - sum) / L[j * n + j];
        }

        // Store column i of inverse
        for (let j = 0; j < n; j++) {
            invArray[j * n + i] = x[j];
        }
    }

    return { inverse, determinant };
}
