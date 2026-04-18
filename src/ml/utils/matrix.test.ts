import { describe, it, expect } from 'vitest';
import { calculateInverseAndDeterminant, getMatrixFromArray } from './matrix';

describe('calculateInverseAndDeterminant', () => {
    it('computes correctly for 2x2 identity matrix', () => {
        const matrix = getMatrixFromArray([
            [1, 0],
            [0, 1],
        ]);

        const { inverse, determinant } = calculateInverseAndDeterminant(matrix);

        expect(determinant).toBeCloseTo(1, 5);
        expect(inverse.shape).toEqual([2, 2]);
        expect(Array.from(inverse.array)).toEqual([1, 0, 0, 1]);
    });

    it('computes correctly for 3x3 identity matrix', () => {
        const matrix = getMatrixFromArray([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]);

        const { inverse, determinant } = calculateInverseAndDeterminant(matrix);

        expect(determinant).toBeCloseTo(1, 5);
        expect(inverse.shape).toEqual([3, 3]);
        expect(Array.from(inverse.array)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    });

    it('computes correctly for a 2x2 positive-definite matrix', () => {
        // A = [[4, 2], [2, 3]]
        // det(A) = 4*3 - 2*2 = 12 - 4 = 8
        // inv(A) = 1/8 * [[3, -2], [-2, 4]] = [[0.375, -0.25], [-0.25, 0.5]]
        const matrix = getMatrixFromArray([
            [4, 2],
            [2, 3],
        ]);

        const { inverse, determinant } = calculateInverseAndDeterminant(matrix);

        expect(determinant).toBeCloseTo(8, 5);

        const expectedInverse = [0.375, -0.25, -0.25, 0.5];

        inverse.array.forEach((val, i) => {
            expect(val).toBeCloseTo(expectedInverse[i], 5);
        });
    });

    it('computes correctly for a 3x3 positive-definite matrix', () => {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        const matrix = getMatrixFromArray([
            [4, 12, -16],
            [12, 37, -43],
            [-16, -43, 98],
        ]);

        const { inverse, determinant } = calculateInverseAndDeterminant(matrix);

        // Cholesky L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        // L diagonal = 2, 1, 3. Det = (2*1*3)^2 = 36
        expect(determinant).toBeCloseTo(36, 5);

        const expectedInverse = [
            1777 / 36,
            -122 / 9,
            19 / 9,
            -122 / 9,
            34 / 9,
            -5 / 9,
            19 / 9,
            -5 / 9,
            1 / 9,
        ];

        inverse.array.forEach((val, i) => {
            expect(val).toBeCloseTo(expectedInverse[i], 4);
        });
    });

    it('handles epsilon parameter for singular/zero matrices', () => {
        const matrix = getMatrixFromArray([
            [0, 0],
            [0, 0],
        ]);

        const epsilon = 1e-4;
        const { inverse, determinant } = calculateInverseAndDeterminant(matrix, epsilon);

        // Cholesky with epsilon:
        // L[0,0] = sqrt(max(0 - 0, epsilon)) = sqrt(epsilon)
        // L[1,0] = 0
        // L[1,1] = sqrt(max(0 - 0, epsilon)) = sqrt(epsilon)
        // Det = (sqrt(epsilon) * sqrt(epsilon))^2 = epsilon^2
        expect(determinant).toBeCloseTo(epsilon * epsilon, 10);

        // Inverse should be ~ [[1/epsilon, 0], [0, 1/epsilon]]
        const expectedInverse = [1 / epsilon, 0, 0, 1 / epsilon];

        inverse.array.forEach((val, i) => {
            expect(val).toBeCloseTo(expectedInverse[i], 3);
        });
    });
});
