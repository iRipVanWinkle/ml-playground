import { type TypedArray, type MatrixLike, EMPTY_MATRIX_LIKE } from '@/ml/matrix';

export { type MatrixLike, type TypedArray, EMPTY_MATRIX_LIKE };

export function row(matrix: MatrixLike, rowIndex: number): TypedArray {
    return matrix.array.subarray(rowIndex * matrix.shape[1], (rowIndex + 1) * matrix.shape[1]);
}

export function element(matrix: MatrixLike, rowIndex: number, colIndex: number): number {
    return matrix.array[rowIndex * matrix.shape[1] + colIndex];
}

export function transposeMatrix(matrix: MatrixLike): MatrixLike {
    const [rows, cols] = matrix.shape;
    const transposed = matrix.array.slice();
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            transposed[j * rows + i] = matrix.array[i * cols + j];
        }
    }
    return { shape: [cols, rows], array: transposed };
}
