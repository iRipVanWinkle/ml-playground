import { describe, it, expect } from 'vitest';
import { tensor2d } from '@tensorflow/tfjs';
import { L1Regularization } from './l1';

describe('L1Regularization', () => {
    describe('compute', () => {
        it('returns 0 if lambda=0', () => {
            const reg = new L1Regularization(0);
            const theta = tensor2d([[1], [2], [3]]);
            const result = reg.compute(theta).arraySync();
            expect(result).toBe(0);
            theta.dispose();
        });

        it('returns 0 if all weights are zero (ignoring bias)', () => {
            const reg = new L1Regularization(2);
            const theta = tensor2d([[10], [0], [0]]);
            const result = reg.compute(theta).arraySync();
            expect(result).toBe(0);
            theta.dispose();
        });

        it('ignores bias term (first row)', () => {
            const reg = new L1Regularization(1);
            const theta = tensor2d([[100], [3], [-4]]);
            // L1 = 1 * (|3| + |-4|) = 7
            const result = reg.compute(theta).arraySync();
            expect(result).toBeCloseTo(7);
            theta.dispose();
        });

        it('computes correct value for known theta and lambda', () => {
            const reg = new L1Regularization(2);
            const theta = tensor2d([[0], [1], [-2]]);
            // L1 = 2 * (|1| + |-2|) = 2 * 3 = 6
            const result = reg.compute(theta).arraySync();
            expect(result).toBeCloseTo(6);
            theta.dispose();
        });

        it('works for multi-column theta', () => {
            const reg = new L1Regularization(1);
            const theta = tensor2d([
                [0, 0], // bias
                [1, -2], // weights
                [-3, 4],
            ]);
            // L1 = 1 * (|1|+|-2|+|3|+|4|) = 1+2+3+4 = 10
            const result = reg.compute(theta).arraySync();
            expect(result).toBeCloseTo(10);
            theta.dispose();
        });
    });

    describe('gradient', () => {
        it('returns 0 if lambda=0', () => {
            const reg = new L1Regularization(0);
            const theta = tensor2d([[1], [2], [3]]);
            const result = reg.gradient(theta).arraySync();
            expect(result).toEqual([[0], [0], [0]]);
            theta.dispose();
        });

        it('returns 0 for all weights if all weights are zero (ignoring bias)', () => {
            const reg = new L1Regularization(2);
            const theta = tensor2d([[10], [0], [0]]);
            const result = reg.gradient(theta).arraySync();
            expect(result).toEqual([[0], [0], [0]]);
            theta.dispose();
        });

        it('ignores bias term (first row)', () => {
            const reg = new L1Regularization(1);
            const theta = tensor2d([[100], [3], [-4]]);
            // Gradient: [0, sign(3), sign(-4)] * 1 = [0, 1, -1]
            const result = reg.gradient(theta).arraySync();
            expect(result).toEqual([[0], [1], [-1]]);
            theta.dispose();
        });

        it('computes correct gradient for known theta and lambda', () => {
            const reg = new L1Regularization(2);
            const theta = tensor2d([[0], [1], [-2]]);
            // Gradient: [0, sign(1)*2, sign(-2)*2] = [0, 2, -2]
            const result = reg.gradient(theta).arraySync();
            expect(result).toEqual([[0], [2], [-2]]);
            theta.dispose();
        });
    });

    it('disposes resources without error', () => {
        const reg = new L1Regularization(0.5);
        expect(() => reg.dispose()).not.toThrow();
        expect(reg['lambda'].isDisposed).toBeTruthy();
        expect(reg['lambda2D'].isDisposed).toBeTruthy();
        expect(reg['zeros2D'].isDisposed).toBeTruthy();
    });
});
