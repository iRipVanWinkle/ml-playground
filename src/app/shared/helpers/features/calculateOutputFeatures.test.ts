import { describe, it, expect } from 'vitest';
import { calculateOutputFeatures, calculateOutputFeatureLabels } from './calculateOutputFeatures';

describe('calculateOutputFeatures', () => {
    describe('sinusoid type', () => {
        it('should return numFeatures * degree for sinusoid', () => {
            expect(calculateOutputFeatures('sinusoid', 2, 3)).toBe(6);
            expect(calculateOutputFeatures('sinusoid', 3, 4)).toBe(12);
            expect(calculateOutputFeatures('sinusoid', 1, 5)).toBe(5);
        });
    });

    describe('cosinusoid type', () => {
        it('should return numFeatures * degree for cosinusoid', () => {
            expect(calculateOutputFeatures('cosinusoid', 2, 3)).toBe(6);
            expect(calculateOutputFeatures('cosinusoid', 3, 4)).toBe(12);
            expect(calculateOutputFeatures('cosinusoid', 1, 5)).toBe(5);
        });
    });

    describe('fourier type', () => {
        it('should return numFeatures * degree * 2 for fourier (sin + cos)', () => {
            expect(calculateOutputFeatures('fourier', 2, 3)).toBe(12);
            expect(calculateOutputFeatures('fourier', 3, 4)).toBe(24);
            expect(calculateOutputFeatures('fourier', 1, 5)).toBe(10);
        });
    });

    describe('polynomial type', () => {
        it('should return correct count for polynomial degree 2', () => {
            // For 2 features, degree 2: x1^2, x1*x2, x2^2 = 3 combinations
            expect(calculateOutputFeatures('polynomial', 2, 2)).toBe(3);
        });

        it('should return correct count for polynomial degree 3', () => {
            // For 2 features, degree 2-3 combinations
            // degree 2: x1^2, x1*x2, x2^2 = 3
            // degree 3: x1^3, x1^2*x2, x1*x2^2, x2^3 = 4
            // total = 7
            expect(calculateOutputFeatures('polynomial', 3, 2)).toBe(7);
        });
    });

    describe('unknown type', () => {
        it('should return 0 for unknown type', () => {
            expect(calculateOutputFeatures('unknown', 2, 3)).toBe(0);
            expect(calculateOutputFeatures('', 2, 3)).toBe(0);
        });
    });
});

describe('calculateOutputFeatureLabels', () => {
    describe('sinusoid type', () => {
        it('should generate sin labels for degree 2', () => {
            const labels = calculateOutputFeatureLabels('sinusoid', 2, ['x', 'y']);
            expect(labels).toEqual(['sin(x)', 'sin(y)', 'sin(2*x)', 'sin(2*y)']);
        });
    });

    describe('cosinusoid type', () => {
        it('should generate cos labels for degree 2', () => {
            const labels = calculateOutputFeatureLabels('cosinusoid', 2, ['x', 'y']);
            expect(labels).toEqual(['cos(x)', 'cos(y)', 'cos(2*x)', 'cos(2*y)']);
        });
    });

    describe('fourier type', () => {
        it('should generate both sin and cos labels for degree 2', () => {
            const labels = calculateOutputFeatureLabels('fourier', 2, ['x']);
            expect(labels).toEqual(['sin(x)', 'sin(2*x)', 'cos(x)', 'cos(2*x)']);
        });
    });

    describe('polynomial type', () => {
        it('should generate polynomial labels for degree 3 with 2 features', () => {
            const labels = calculateOutputFeatureLabels('polynomial', 3, ['x', 'y']);
            expect(labels).toEqual(['y^2', 'x*y', 'x^2', 'y^3', 'x*y^2', 'x^2*y', 'x^3']);
        });
    });

    describe('unknown type', () => {
        it('should return empty array for unknown type', () => {
            expect(calculateOutputFeatureLabels('unknown', 2, ['x', 'y'])).toEqual([]);
            expect(calculateOutputFeatureLabels('', 2, ['x', 'y'])).toEqual([]);
        });
    });
});
