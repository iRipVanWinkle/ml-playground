import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import { getMatrixFromTensor, type MatrixLike } from '@/ml/matrix';

/**
 * Gets a matrix from a tensor, safely handling undefined values.
 * @param tensor - The tensor to get the matrix from, or undefined.
 * @returns A matrix, or undefined if tensor is undefined.
 */
export async function getSafeMatrixFromTensor(tensor: Tensor2D): Promise<MatrixLike>;
export async function getSafeMatrixFromTensor(
    tensor: Tensor2D | undefined,
): Promise<MatrixLike | undefined>;
export async function getSafeMatrixFromTensor(tensor?: Tensor2D): Promise<MatrixLike | undefined> {
    if (tensor === undefined) {
        return undefined;
    }

    return getMatrixFromTensor(tensor);
}

/**
 * Gets an array from a tensor, safely handling undefined values.
 * @param tensor - The tensor to get the array from, or undefined.
 * @returns An array, or undefined if tensor is undefined.
 */
export async function getSafeTensorArray(tensor: Tensor2D): Promise<number[][]>;
export async function getSafeTensorArray(
    tensor: Tensor2D | undefined,
): Promise<number[][] | undefined>;
export async function getSafeTensorArray(tensor?: Tensor2D): Promise<number[][] | undefined> {
    if (tensor === undefined) {
        return undefined;
    }

    return tensor.array();
}

/**
 * Gets a scalar value from a tensor, safely handling undefined values.
 * @param tensor - The scalar tensor to get the value from, or undefined.
 * @returns A number, or undefined if tensor is undefined.
 */
export async function getSafeTensorValue(tensor: Scalar): Promise<number>;
export async function getSafeTensorValue(tensor: Scalar | undefined): Promise<number | undefined>;
export async function getSafeTensorValue(tensor?: Scalar): Promise<number | undefined> {
    if (tensor === undefined) {
        return undefined;
    }

    const data = await tensor.data();
    return data[0];
}
