import { describe, it, expect } from 'vitest';
import {
    zerosInitializer,
    onesInitializer,
    constantInitializer,
    uniformInitializer,
    normalInitializer,
    xavierUniformInitializer,
    xavierNormalInitializer,
    heUniformInitializer,
    heNormalInitializer,
} from './initializers';

describe('Theta Initializers', () => {
    describe('zerosInitializer', () => {
        it('should create tensor with all zeros', () => {
            const initializer = zerosInitializer();
            const theta = initializer([2, 4], true);

            expect(theta.shape).toEqual([3, 4]); // shape[0] + 1 for bias
            const values = theta.arraySync();
            values.forEach((row) => {
                row.forEach((val) => {
                    expect(val).toBe(0);
                });
            });

            theta.dispose();
        });

        it('should create correct shape without bias', () => {
            const initializer = zerosInitializer();
            const theta = initializer([2, 3], false);

            expect(theta.shape).toEqual([2, 3]);
            const values = theta.arraySync();
            values.forEach((row) => {
                row.forEach((val) => {
                    expect(val).toBe(0);
                });
            });

            theta.dispose();
        });
    });

    describe('onesInitializer', () => {
        it('should create tensor with ones and bias row of zeros', () => {
            const initializer = onesInitializer();
            const theta = initializer([2, 4], true);

            expect(theta.shape).toEqual([3, 4]); // shape[0] + 1 for bias
            const values = theta.arraySync();
            // First row (bias) should be zeros
            values[0].forEach((val) => {
                expect(val).toBe(0);
            });
            // Rest should be ones
            for (let i = 1; i < values.length; i++) {
                values[i].forEach((val) => {
                    expect(val).toBe(1);
                });
            }

            theta.dispose();
        });

        it('should create correct shape without bias', () => {
            const initializer = onesInitializer();
            const theta = initializer([2, 3], false);

            expect(theta.shape).toEqual([2, 3]);
            const values = theta.arraySync();
            values.forEach((row) => {
                row.forEach((val) => {
                    expect(val).toBe(1);
                });
            });

            theta.dispose();
        });
    });

    describe('constantInitializer', () => {
        it('should create tensor with constant value and bias row of zeros', () => {
            const constant = 5.5;
            const initializer = constantInitializer(constant);
            const theta = initializer([2, 3], true);

            expect(theta.shape).toEqual([3, 3]);
            const values = theta.arraySync();
            // First row (bias) should be zeros
            values[0].forEach((val) => {
                expect(val).toBe(0);
            });
            // Rest should be constant
            for (let i = 1; i < values.length; i++) {
                values[i].forEach((val) => {
                    expect(val).toBe(constant);
                });
            }

            theta.dispose();
        });

        it('should create correct shape without bias', () => {
            const constant = -2.5;
            const initializer = constantInitializer(constant);
            const theta = initializer([2, 3], false);

            expect(theta.shape).toEqual([2, 3]);
            const values = theta.arraySync();
            values.forEach((row) => {
                row.forEach((val) => {
                    expect(val).toBe(constant);
                });
            });

            theta.dispose();
        });
    });

    describe('uniformInitializer', () => {
        it('should generate values within specified range', () => {
            const min = -1;
            const max = 1;
            const initializer = uniformInitializer(min, max);
            const theta = initializer([10, 5], true);

            expect(theta.shape).toEqual([11, 5]);
            const values = theta.arraySync();
            // First row (bias) should be zeros
            values[0].forEach((val) => {
                expect(val).toBe(0);
            });
            // Rest should be in the specified range
            for (let i = 1; i < values.length; i++) {
                values[i].forEach((val) => {
                    expect(val).toBeGreaterThanOrEqual(min);
                    expect(val).toBeLessThan(max);
                });
            }

            theta.dispose();
        });

        it('should handle negative ranges and work without bias', () => {
            const min = -5;
            const max = -1;
            const initializer = uniformInitializer(min, max);
            const theta = initializer([3, 4], false);

            expect(theta.shape).toEqual([3, 4]);
            const values = theta.arraySync();

            values.forEach((row) => {
                row.forEach((val) => {
                    expect(val).toBeGreaterThanOrEqual(min);
                    expect(val).toBeLessThan(max);
                });
            });

            theta.dispose();
        });
    });

    describe('normalInitializer', () => {
        it('should generate values with approximately correct mean and stddev', () => {
            const mean = 0;
            const stddev = 1;
            const initializer = normalInitializer(mean, stddev);
            const theta = initializer([100, 5], true);

            expect(theta.shape).toEqual([101, 5]);
            const values = theta.arraySync();
            // Flatten the 2D array to 1D for statistics calculation
            const flatValues = values.flat();
            const totalElements = flatValues.length;

            const sampleMean = flatValues.reduce((sum, val) => sum + val, 0) / totalElements;
            const sampleVariance =
                flatValues.reduce((sum, val) => sum + (val - sampleMean) ** 2, 0) /
                (totalElements - 1);
            const sampleStddev = Math.sqrt(sampleVariance);

            expect(sampleMean).toBeCloseTo(mean, 0);
            expect(sampleStddev).toBeCloseTo(stddev, 0);

            theta.dispose();
        });

        it('should work with custom mean and stddev', () => {
            const mean = 5;
            const stddev = 2;
            const initializer = normalInitializer(mean, stddev);
            const theta = initializer([50, 3], false);

            expect(theta.shape).toEqual([50, 3]);
            const values = Array.from(theta.dataSync());

            const sampleMean = values.reduce((sum, val) => sum + val, 0) / values.length;
            expect(sampleMean).toBeCloseTo(mean, 0);

            theta.dispose();
        });
    });

    describe('xavierUniformInitializer', () => {
        it('should generate values within calculated limit', () => {
            const initializer = xavierUniformInitializer();
            const shape: [number, number] = [4, 8];
            const theta = initializer(shape, true);

            expect(theta.shape).toEqual([5, 8]); // 4 + 1 for bias
            const limit = Math.sqrt(6 / (shape[0] + shape[1]));
            const values = theta.arraySync() as number[][];
            // First row (bias) should be zeros
            values[0].forEach((val) => {
                expect(val).toBe(0);
            });
            // Rest should be within limit
            for (let i = 1; i < values.length; i++) {
                values[i].forEach((val) => {
                    expect(val).toBeGreaterThanOrEqual(-limit);
                    expect(val).toBeLessThan(limit);
                });
            }

            theta.dispose();
        });

        it('should work without bias', () => {
            const initializer = xavierUniformInitializer();
            const shape: [number, number] = [3, 5];
            const theta = initializer(shape, false);

            expect(theta.shape).toEqual(shape);
            const limit = Math.sqrt(6 / (shape[0] + shape[1]));
            const values = Array.from(theta.dataSync());

            values.forEach((val) => {
                expect(Math.abs(val)).toBeLessThan(limit);
            });

            theta.dispose();
        });
    });

    describe('xavierNormalInitializer', () => {
        it('should generate values with correct stddev', () => {
            const initializer = xavierNormalInitializer();
            const shape: [number, number] = [4, 8];
            const theta = initializer(shape, true);

            expect(theta.shape).toEqual([5, 8]);
            const expectedStddev = Math.sqrt(2 / (shape[0] + shape[1]));
            const values = Array.from(theta.dataSync());

            const sampleMean = values.reduce((sum, val) => sum + val, 0) / values.length;
            const sampleVariance =
                values.reduce((sum, val) => sum + (val - sampleMean) ** 2, 0) / (values.length - 1);
            const sampleStddev = Math.sqrt(sampleVariance);

            expect(sampleMean).toBeCloseTo(0, 0);
            expect(sampleStddev).toBeCloseTo(expectedStddev, 0);

            theta.dispose();
        });

        it('should work without bias', () => {
            const initializer = xavierNormalInitializer();
            const shape: [number, number] = [6, 4];
            const theta = initializer(shape, false);

            expect(theta.shape).toEqual([6, 4]);
            const values = Array.from(theta.dataSync());

            expect(values.length).toBe(24);

            theta.dispose();
        });
    });

    describe('heUniformInitializer', () => {
        it('should generate values within calculated limit based on fan-in', () => {
            const initializer = heUniformInitializer();
            const shape: [number, number] = [4, 8];
            const theta = initializer(shape, true);

            expect(theta.shape).toEqual([5, 8]);
            const limit = Math.sqrt(6 / shape[0]);
            const values = theta.arraySync() as number[][];
            // First row (bias) should be zeros
            values[0].forEach((val) => {
                expect(val).toBe(0);
            });
            // Rest should be within limit
            for (let i = 1; i < values.length; i++) {
                values[i].forEach((val) => {
                    expect(val).toBeGreaterThanOrEqual(-limit);
                    expect(val).toBeLessThan(limit);
                });
            }

            theta.dispose();
        });

        it('should work without bias', () => {
            const initializer = heUniformInitializer();
            const shape: [number, number] = [10, 3];
            const theta = initializer(shape, false);

            expect(theta.shape).toEqual([10, 3]);
            const limit = Math.sqrt(6 / shape[0]);
            const values = Array.from(theta.dataSync());

            values.forEach((val) => {
                expect(Math.abs(val)).toBeLessThan(limit);
            });

            theta.dispose();
        });
    });

    describe('heNormalInitializer', () => {
        it('should generate values with correct stddev based on fan-in', () => {
            const initializer = heNormalInitializer();
            const shape: [number, number] = [4, 8];
            const theta = initializer(shape, true);

            expect(theta.shape).toEqual([5, 8]);
            const expectedStddev = Math.sqrt(2 / shape[0]);
            const values = Array.from(theta.dataSync());

            const sampleMean = values.reduce((sum, val) => sum + val, 0) / values.length;
            const sampleVariance =
                values.reduce((sum, val) => sum + (val - sampleMean) ** 2, 0) / (values.length - 1);
            const sampleStddev = Math.sqrt(sampleVariance);

            expect(sampleMean).toBeCloseTo(0, 0);
            expect(sampleStddev).toBeCloseTo(expectedStddev, 0);

            theta.dispose();
        });

        it('should work without bias', () => {
            const initializer = heNormalInitializer();
            const shape: [number, number] = [8, 6];
            const theta = initializer(shape, false);

            expect(theta.shape).toEqual([8, 6]);
            const values = Array.from(theta.dataSync());

            expect(values.length).toBe(48);

            theta.dispose();
        });
    });
});
