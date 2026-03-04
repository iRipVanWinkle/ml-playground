import { Matrix, type MatrixLike } from '../matrix';
import { EPSILON } from '../constants';

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
        for (let j = 0; j <= i; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }

            if (i === j) {
                L[i * n + j] = Math.sqrt(Math.max(array[i * n + i] - sum, epsilon));
            } else {
                L[i * n + j] = (array[i * n + j] - sum) / L[j * n + j];
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

    for (let i = 0; i < n; i++) {
        const b = new Float32Array(n);
        b[i] = 1;

        // Forward substitution: solve Ly = b
        const y = new Float32Array(n);
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) {
                sum += L[j * n + k] * y[k];
            }
            y[j] = (b[j] - sum) / L[j * n + j];
        }

        // Backward substitution: solve Lᵀx = y
        const x = new Float32Array(n);
        for (let j = n - 1; j >= 0; j--) {
            let sum = 0;
            for (let k = j + 1; k < n; k++) {
                sum += L[k * n + j] * x[k];
            }
            x[j] = (y[j] - sum) / L[j * n + j];
        }

        // Store column i of inverse
        for (let j = 0; j < n; j++) {
            inverse.array[j * n + i] = x[j];
        }
    }

    return { inverse, determinant };
}
