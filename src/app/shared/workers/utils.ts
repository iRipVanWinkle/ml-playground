import type { Scalar, Tensor1D, Tensor2D } from '@tensorflow/tfjs';
import { getMatrixFromTensor, type MatrixLike, type TypedArray } from '@/ml/matrix';

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
 * Gets an data from a tensor, safely handling undefined values.
 * @param tensor - The tensor to get the array from, or undefined.
 * @returns A typedarray, or undefined if tensor is undefined.
 */
export async function getSafeTensorTypedArray(tensor: Tensor1D): Promise<TypedArray>;
export async function getSafeTensorTypedArray(
    tensor: Tensor1D | undefined,
): Promise<TypedArray | undefined>;
export async function getSafeTensorTypedArray(tensor?: Tensor1D): Promise<TypedArray | undefined> {
    if (tensor === undefined) {
        return undefined;
    }

    return tensor.data();
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

interface DisposableValue {
    dispose: () => void;
}

type MappedValues<T extends object, IsPartial extends 'required' | 'partial' = 'required'> = {
    [P in keyof T]: IsPartial extends 'partial' ? T[P] | undefined : T[P];
};

/**
 * Creates a container for tensors that can be disposed of collectively.
 * @returns An object containing tensors with a dispose method to free resources.
 */
export function createTensorContainer<
    T extends object,
    IsPartial extends 'required' | 'partial' = 'required',
>(): MappedValues<T, IsPartial> & DisposableValue {
    const container = {
        dispose(): void {
            for (const key in container) {
                const tensor = container[key as keyof typeof container];
                if (isDisposable(tensor)) {
                    tensor.dispose();
                }
            }
        },
    } as MappedValues<T, IsPartial> & DisposableValue;

    return container as MappedValues<T, IsPartial> & DisposableValue;
}

function isDisposable(value: unknown): value is DisposableValue {
    return (
        !!value &&
        typeof value === 'object' &&
        'dispose' in value &&
        typeof value.dispose === 'function'
    );
}
