import { describe, it, expect } from 'vitest';
import {
    calculateMean,
    calculateVariance,
    calculateCovarianceMatrix,
    calculateFullGaussianLogPdf,
    calculateDiagonalGaussianLogPdf,
} from './statistics';

describe('statistics utils', () => {
    describe('calculateMean', () => {
        it('should calculate the mean of all samples correctly', () => {
            const X = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ];

            // Expected means:
            // Feature 0: (1+4+7)/3 = 4
            // Feature 1: (2+5+8)/3 = 5
            // Feature 2: (3+6+9)/3 = 6
            const result = calculateMean(X, 3);

            expect(result).toBeInstanceOf(Float32Array);
            expect(Array.from(result)).toEqual([4, 5, 6]);
        });

        it('should calculate the mean using specific indices correctly', () => {
            const X = [
                [1, 10, 100],
                [4, 40, 400], // index 1
                [7, 70, 700],
                [2, 20, 200], // index 3
            ];

            const indices = [1, 3];

            // Expected means for samples at index 1 and 3:
            // Feature 0: (4+2)/2 = 3
            // Feature 1: (40+20)/2 = 30
            // Feature 2: (400+200)/2 = 300
            const result = calculateMean(X, 3, indices);

            expect(result).toBeInstanceOf(Float32Array);
            expect(Array.from(result)).toEqual([3, 30, 300]);
        });

        it('should handle floating point numbers correctly', () => {
            const X = [
                [1.5, 2.5],
                [3.5, 4.5],
            ];

            const result = calculateMean(X, 2);

            expect(result[0]).toBeCloseTo(2.5);
            expect(result[1]).toBeCloseTo(3.5);
        });

        it('should return zeros for empty dataset', () => {
            const X: number[][] = [];
            const result = calculateMean(X, 2);

            expect(Array.from(result)).toEqual([0, 0]);
        });

        it('should return zeros when indices array is empty', () => {
            const X = [
                [1, 2],
                [3, 4],
            ];
            const result = calculateMean(X, 2, []);

            expect(Array.from(result)).toEqual([0, 0]);
        });
    });

    describe('calculateVariance', () => {
        it('should calculate the variance of all samples correctly', () => {
            const X = [
                [2, 4],
                [4, 4],
                [4, 5],
                [4, 5],
                [6, 7],
            ];

            // Mean 0: (2+4+4+4+6)/5 = 4
            // Mean 1: (4+4+5+5+7)/5 = 5
            const means = new Float32Array([4, 5]);

            // Var 0: ((2-4)^2 + (4-4)^2 + (4-4)^2 + (4-4)^2 + (6-4)^2) / 5
            // = (4 + 0 + 0 + 0 + 4) / 5 = 1.6
            // Var 1: ((4-5)^2 + (4-5)^2 + (5-5)^2 + (5-5)^2 + (7-5)^2) / 5
            // = (1 + 1 + 0 + 0 + 4) / 5 = 1.2

            const result = calculateVariance(X, means, 2);

            expect(result).toBeInstanceOf(Float32Array);
            expect(result[0]).toBeCloseTo(1.6);
            expect(result[1]).toBeCloseTo(1.2);
        });

        it('should calculate the variance using specific indices correctly', () => {
            const X = [
                [100, 100],
                [2, 4], // index 1
                [100, 100],
                [4, 4], // index 3
                [4, 5], // index 4
                [4, 5], // index 5
                [100, 100],
                [6, 7], // index 7
            ];
            const indices = [1, 3, 4, 5, 7];
            const means = new Float32Array([4, 5]);

            const result = calculateVariance(X, means, 2, indices);

            expect(result[0]).toBeCloseTo(1.6);
            expect(result[1]).toBeCloseTo(1.2);
        });

        it('should return zeros for empty dataset', () => {
            const X: number[][] = [];
            const result = calculateVariance(X, new Float32Array([0, 0]), 2);
            expect(Array.from(result)).toEqual([0, 0]);
        });
    });

    describe('calculateCovarianceMatrix', () => {
        it('should calculate the full covariance matrix correctly', () => {
            // Data array:
            // x y
            // 2 4
            // 4 4
            // 4 5
            // 4 5
            // 6 7
            const X = [
                [2, 4],
                [4, 4],
                [4, 5],
                [4, 5],
                [6, 7],
            ];
            const means = new Float32Array([4, 5]);

            // Var(X) = 1.6
            // Var(Y) = 1.2
            // Cov(X,Y) = E[(X-ux)(Y-uy)]
            // = [ (2-4)(4-5) + (4-4)(4-5) + (4-4)(5-5) + (4-4)(5-5) + (6-4)(7-5) ] / 5
            // = [ 2 + 0 + 0 + 0 + 4 ] / 5 = 1.2
            //
            // Covariance Matrix:
            // [ 1.6  1.2 ]
            // [ 1.2  1.2 ]

            const result = calculateCovarianceMatrix(X, means, 2);

            expect(result.shape).toEqual([2, 2]);
            expect(result.array.length).toBe(4); // 2x2 flat

            expect(result.array[0]).toBeCloseTo(1.6); // [0][0] Var(X)
            expect(result.array[1]).toBeCloseTo(1.2); // [0][1] Cov(X,Y)
            expect(result.array[2]).toBeCloseTo(1.2); // [1][0] Cov(Y,X)
            expect(result.array[3]).toBeCloseTo(1.2); // [1][1] Var(Y)
        });

        it('should calculate the covariance matrix using indices correctly', () => {
            const X = [
                [100, 100],
                [2, 4], // index 1
                [100, 100],
                [4, 4], // index 3
                [4, 5], // index 4
                [4, 5], // index 5
                [100, 100],
                [6, 7], // index 7
            ];
            const indices = [1, 3, 4, 5, 7];
            const means = new Float32Array([4, 5]);

            const result = calculateCovarianceMatrix(X, means, 2, indices);

            expect(result.shape).toEqual([2, 2]);
            expect(result.array[0]).toBeCloseTo(1.6); // [0][0] Var(X)
            expect(result.array[1]).toBeCloseTo(1.2); // [0][1] Cov(X,Y)
            expect(result.array[2]).toBeCloseTo(1.2); // [1][0] Cov(Y,X)
            expect(result.array[3]).toBeCloseTo(1.2); // [1][1] Var(Y)
        });

        it('should return zeros for empty dataset', () => {
            const X: number[][] = [];
            const result = calculateCovarianceMatrix(X, new Float32Array([0, 0]), 2);
            expect(Array.from(result.array)).toEqual([0, 0, 0, 0]);
        });
    });

    describe('calculateDiagonalGaussianLogPdf', () => {
        it('should compute correct log-PDF for standard normal', () => {
            // N(0,1): log-PDF at x=0 should be -0.5*log(2π) ≈ -0.9189
            const result = calculateDiagonalGaussianLogPdf(
                [0],
                new Float32Array([0]),
                new Float32Array([1]),
            );
            expect(result).toBeCloseTo(-0.5 * Math.log(2 * Math.PI));
        });

        it('should compute correct log-PDF for multiple features', () => {
            const sample = [1, 2];
            const means = new Float32Array([1, 2]);
            const variances = new Float32Array([1, 1]);

            // At the mean, each feature contributes -0.5*log(2π)
            const expected = 2 * (-0.5 * Math.log(2 * Math.PI));
            const result = calculateDiagonalGaussianLogPdf(sample, means, variances);
            expect(result).toBeCloseTo(expected);
        });

        it('should return lower log-PDF further from the mean', () => {
            const means = new Float32Array([0]);
            const variances = new Float32Array([1]);

            const atMean = calculateDiagonalGaussianLogPdf([0], means, variances);
            const awayFromMean = calculateDiagonalGaussianLogPdf([3], means, variances);

            expect(awayFromMean).toBeLessThan(atMean);
        });
    });

    describe('calculateFullGaussianLogPdf', () => {
        it('should compute correct log-PDF for identity covariance at mean', () => {
            // 2D standard normal at origin
            const sample = [0, 0];
            const means = new Float32Array([0, 0]);
            const covInverse = {
                array: new Float32Array([1, 0, 0, 1]),
                shape: [2, 2] as [number, number],
            };
            const det = 1;

            // log N(0|0,I) = -0.5 * (2*log(2π) + log(1) + 0) = -log(2π)
            const expected = -Math.log(2 * Math.PI);
            const result = calculateFullGaussianLogPdf(sample, means, covInverse, det);
            expect(result).toBeCloseTo(expected);
        });

        it('should return lower log-PDF further from the mean', () => {
            const means = new Float32Array([0, 0]);
            const covInverse = {
                array: new Float32Array([1, 0, 0, 1]),
                shape: [2, 2] as [number, number],
            };
            const det = 1;

            const atMean = calculateFullGaussianLogPdf([0, 0], means, covInverse, det);
            const awayFromMean = calculateFullGaussianLogPdf([3, 3], means, covInverse, det);

            expect(awayFromMean).toBeLessThan(atMean);
        });
    });
});
