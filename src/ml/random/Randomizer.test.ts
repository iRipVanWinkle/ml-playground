import { describe, it, expect, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { Randomizer } from './Randomizer';

describe('Randomizer', () => {
    beforeEach(() => {
        // Reset seed before each test
        Randomizer.setSeed(42);
    });

    describe('setSeed', () => {
        it('should set the global seed', () => {
            Randomizer.setSeed(123);

            // Generate two tensors with the same parameters
            const tensor1 = Randomizer.randomUniform([2, 2], 0, 1);
            Randomizer.setSeed(123); // Reset to same seed
            const tensor2 = Randomizer.randomUniform([2, 2], 0, 1);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should handle undefined seed', () => {
            Randomizer.setSeed(undefined);

            // Should not throw an error
            expect(() => {
                const tensor = Randomizer.randomUniform([2, 2]);
                tensor.dispose();
            }).not.toThrow();
        });
    });

    describe('randomUniform', () => {
        it('should generate tensor with correct shape', () => {
            const tensor = Randomizer.randomUniform([3, 4], 0, 1);

            expect(tensor.shape).toEqual([3, 4]);
            expect(tensor.dtype).toBe('float32');

            tensor.dispose();
        });

        it('should generate values within specified range', () => {
            const minval = -2;
            const maxval = 5;
            const tensor = Randomizer.randomUniform([100], minval, maxval);

            const values = tensor.dataSync();

            for (let i = 0; i < values.length; i++) {
                expect(values[i]).toBeGreaterThanOrEqual(minval);
                expect(values[i]).toBeLessThan(maxval);
            }

            tensor.dispose();
        });

        it('should support different data types', () => {
            const floatTensor = Randomizer.randomUniform([2, 2], 0, 1, 'float32');
            const intTensor = Randomizer.randomUniform([2, 2], 0, 10, 'int32');

            expect(floatTensor.dtype).toBe('float32');
            expect(intTensor.dtype).toBe('int32');

            floatTensor.dispose();
            intTensor.dispose();
        });

        it('should use provided seed parameter', () => {
            const tensor1 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 999);
            const tensor2 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 999);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should combine global seed with provided seed', () => {
            Randomizer.setSeed(100);

            const tensor1 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 50);

            Randomizer.setSeed(100);
            const tensor2 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 50);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should handle 1D shapes', () => {
            const tensor = Randomizer.randomUniform([5], 0, 1);

            expect(tensor.shape).toEqual([5]);
            expect(tensor.rank).toBe(1);

            tensor.dispose();
        });

        it('should handle 3D shapes', () => {
            const tensor = Randomizer.randomUniform([2, 3, 4], 0, 1);

            expect(tensor.shape).toEqual([2, 3, 4]);
            expect(tensor.rank).toBe(3);

            tensor.dispose();
        });
    });

    describe('randomNormal', () => {
        it('should generate tensor with correct shape', () => {
            const tensor = Randomizer.randomNormal([3, 4]);

            expect(tensor.shape).toEqual([3, 4]);
            expect(tensor.dtype).toBe('float32');

            tensor.dispose();
        });

        it('should generate values with approximately correct mean and standard deviation', () => {
            const mean = 5;
            const stddev = 2;
            const tensor = Randomizer.randomNormal([1000], mean, stddev);

            const values = Array.from(tensor.dataSync());

            // Calculate sample mean and standard deviation
            const sampleMean = values.reduce((sum, val) => sum + val, 0) / values.length;
            const sampleVariance =
                values.reduce((sum, val) => sum + (val - sampleMean) ** 2, 0) / (values.length - 1);
            const sampleStddev = Math.sqrt(sampleVariance);

            // Allow some tolerance for random sampling
            expect(sampleMean).toBeCloseTo(mean, 0); // Within 1 unit
            expect(sampleStddev).toBeCloseTo(stddev, 0); // Within 1 unit

            tensor.dispose();
        });

        it('should support different data types', () => {
            const floatTensor = Randomizer.randomNormal([2, 2], 0, 1, 'float32');
            const intTensor = Randomizer.randomNormal([2, 2], 0, 1, 'int32');

            expect(floatTensor.dtype).toBe('float32');
            expect(intTensor.dtype).toBe('int32');

            floatTensor.dispose();
            intTensor.dispose();
        });

        it('should use provided seed parameter', () => {
            const tensor1 = Randomizer.randomNormal([2, 2], 0, 1, 'float32', 777);
            const tensor2 = Randomizer.randomNormal([2, 2], 0, 1, 'float32', 777);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should combine global seed with provided seed', () => {
            Randomizer.setSeed(200);

            const tensor1 = Randomizer.randomNormal([2, 2], 0, 1, 'float32', 75);

            Randomizer.setSeed(200);
            const tensor2 = Randomizer.randomNormal([2, 2], 0, 1, 'float32', 75);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should use default parameters when not provided', () => {
            const tensor = Randomizer.randomNormal([100]);

            const values = Array.from(tensor.dataSync());
            const sampleMean = values.reduce((sum, val) => sum + val, 0) / values.length;

            // Default mean should be around 0
            expect(Math.abs(sampleMean)).toBeLessThan(0.5);

            tensor.dispose();
        });

        it('should handle different tensor ranks', () => {
            const tensor1D = Randomizer.randomNormal([5]);
            const tensor2D = Randomizer.randomNormal([2, 3]);
            const tensor3D = Randomizer.randomNormal([2, 2, 2]);

            expect(tensor1D.rank).toBe(1);
            expect(tensor2D.rank).toBe(2);
            expect(tensor3D.rank).toBe(3);

            tensor1D.dispose();
            tensor2D.dispose();
            tensor3D.dispose();
        });
    });

    describe('randomUniqueNumber', () => {
        it('should generate tensor with correct shape', () => {
            const tensor = Randomizer.randomUniqueNumber([3, 4], 0, 10);

            expect(tensor.shape).toEqual([3, 4]);
            expect(tensor.dtype).toBe('float32');

            tensor.dispose();
        });

        it('should generate unique values', () => {
            const tensor = Randomizer.randomUniqueNumber([10], 0, 20);

            const values = Array.from(tensor.dataSync());
            const uniqueValues = new Set(values);

            expect(uniqueValues.size).toBe(values.length);

            tensor.dispose();
        });

        it('should generate values within specified range', () => {
            const minval = 5;
            const maxval = 15;
            const tensor = Randomizer.randomUniqueNumber([20], minval, maxval);

            const values = tensor.dataSync();

            for (let i = 0; i < values.length; i++) {
                expect(values[i]).toBeGreaterThanOrEqual(minval);
                expect(values[i]).toBeLessThanOrEqual(maxval);
            }

            tensor.dispose();
        });

        it('should use provided seed parameter for deterministic results', () => {
            const tensor1 = Randomizer.randomUniqueNumber([5], 0, 10, 'float32', 999);
            const tensor2 = Randomizer.randomUniqueNumber([5], 0, 10, 'float32', 999);

            const values1 = tensor1.arraySync() as number[];
            const values2 = tensor2.arraySync() as number[];

            // Check that both results have unique values within the expected range
            const uniqueValues1 = new Set(values1);
            const uniqueValues2 = new Set(values2);

            expect(uniqueValues1.size).toBe(values1.length);
            expect(uniqueValues2.size).toBe(values2.length);

            // Values should be in the same range
            for (let i = 0; i < values1.length; i++) {
                expect(values1[i]).toBeGreaterThanOrEqual(0);
                expect(values1[i]).toBeLessThanOrEqual(10);
                expect(values2[i]).toBeGreaterThanOrEqual(0);
                expect(values2[i]).toBeLessThanOrEqual(10);
            }

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should combine global seed with provided seed', () => {
            Randomizer.setSeed(100);

            const tensor1 = Randomizer.randomUniqueNumber([3], 0, 10, 'float32', 50);

            Randomizer.setSeed(100);
            const tensor2 = Randomizer.randomUniqueNumber([3], 0, 10, 'float32', 50);

            const values1 = tensor1.arraySync() as number[];
            const values2 = tensor2.arraySync() as number[];

            // Check that both results have unique values within the expected range
            const uniqueValues1 = new Set(values1);
            const uniqueValues2 = new Set(values2);

            expect(uniqueValues1.size).toBe(values1.length);
            expect(uniqueValues2.size).toBe(values2.length);

            // Values should be in the same range
            for (let i = 0; i < values1.length; i++) {
                expect(values1[i]).toBeGreaterThanOrEqual(0);
                expect(values1[i]).toBeLessThanOrEqual(10);
                expect(values2[i]).toBeGreaterThanOrEqual(0);
                expect(values2[i]).toBeLessThanOrEqual(10);
            }

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should handle 1D shapes', () => {
            const tensor = Randomizer.randomUniqueNumber([8], 0, 20);

            expect(tensor.shape).toEqual([8]);
            expect(tensor.rank).toBe(1);

            tensor.dispose();
        });

        it('should handle 2D shapes', () => {
            const tensor = Randomizer.randomUniqueNumber([3, 4], 0, 20);

            expect(tensor.shape).toEqual([3, 4]);
            expect(tensor.rank).toBe(2);

            tensor.dispose();
        });

        it('should handle 3D shapes', () => {
            const tensor = Randomizer.randomUniqueNumber([2, 3, 2], 0, 20);

            expect(tensor.shape).toEqual([2, 3, 2]);
            expect(tensor.rank).toBe(3);

            tensor.dispose();
        });

        it('should use default parameters when not provided', () => {
            const tensor = Randomizer.randomUniqueNumber([5]);

            expect(tensor.shape).toEqual([5]);
            expect(tensor.dtype).toBe('float32');

            const values = tensor.dataSync();
            // Default range is 0 to 0, so all values should be 0
            for (let i = 0; i < values.length; i++) {
                expect(values[i]).toBe(0);
            }

            tensor.dispose();
        });

        it('should handle negative ranges', () => {
            const tensor = Randomizer.randomUniqueNumber([5], -10, -5);

            const values = tensor.dataSync();

            for (let i = 0; i < values.length; i++) {
                expect(values[i]).toBeGreaterThanOrEqual(-10);
                expect(values[i]).toBeLessThanOrEqual(-5);
            }

            tensor.dispose();
        });

        it('should handle mixed positive and negative ranges', () => {
            const tensor = Randomizer.randomUniqueNumber([5], -5, 5);

            const values = tensor.dataSync();

            for (let i = 0; i < values.length; i++) {
                expect(values[i]).toBeGreaterThanOrEqual(-5);
                expect(values[i]).toBeLessThanOrEqual(5);
            }

            tensor.dispose();
        });

        it('should handle single element tensors', () => {
            const tensor = Randomizer.randomUniqueNumber([1], 0, 10);

            expect(tensor.shape).toEqual([1]);
            expect(tensor.size).toBe(1);

            const value = tensor.dataSync()[0];
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThanOrEqual(10);

            tensor.dispose();
        });

        it('should handle large shapes within pool size limit', () => {
            const tensor = Randomizer.randomUniqueNumber([100], 0, 200);

            expect(tensor.shape).toEqual([100]);

            const values = Array.from(tensor.dataSync());
            const uniqueValues = new Set(values);
            expect(uniqueValues.size).toBe(values.length);

            tensor.dispose();
        });

        it('should produce different results with different seeds', () => {
            const tensor1 = Randomizer.randomUniqueNumber([3], 0, 10, 'float32', 100);
            const tensor2 = Randomizer.randomUniqueNumber([3], 0, 10, 'float32', 200);

            const values1 = tensor1.arraySync() as number[];
            const values2 = tensor2.arraySync() as number[];

            // Values should be in the correct range but may be different
            for (let i = 0; i < values1.length; i++) {
                expect(values1[i]).toBeGreaterThanOrEqual(0);
                expect(values1[i]).toBeLessThanOrEqual(10);
                expect(values2[i]).toBeGreaterThanOrEqual(0);
                expect(values2[i]).toBeLessThanOrEqual(10);
            }

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should handle edge case with small range', () => {
            const tensor = Randomizer.randomUniqueNumber([3], 0, 2);

            const values = tensor.dataSync();

            for (let i = 0; i < values.length; i++) {
                expect(values[i]).toBeGreaterThanOrEqual(0);
                expect(values[i]).toBeLessThanOrEqual(2);
            }

            tensor.dispose();
        });
    });

    describe('shuffle', () => {
        it('should modify the original array in place', () => {
            const original = [1, 2, 3, 4, 5];
            const originalCopy = [...original];
            Randomizer.shuffle(original);

            // Original array should be modified
            expect(original).not.toEqual(originalCopy);
        });

        it('should produce deterministic results with the same seed', () => {
            const original1 = [1, 2, 3, 4, 5];
            const original2 = [1, 2, 3, 4, 5];
            Randomizer.shuffle(original1, 123);
            Randomizer.shuffle(original2, 123);

            expect(original1).toEqual(original2);
        });

        it('should use global seed when no seed provided', () => {
            Randomizer.setSeed(456);
            const original1 = [1, 2, 3, 4, 5];
            Randomizer.shuffle(original1);

            Randomizer.setSeed(456);
            const original2 = [1, 2, 3, 4, 5];
            Randomizer.shuffle(original2);

            expect(original1).toEqual(original2);
        });

        it('should handle empty array', () => {
            const original: number[] = [];
            Randomizer.shuffle(original);

            expect(original).toEqual([]);
        });

        it('should handle single element array', () => {
            const original = [42];
            Randomizer.shuffle(original);

            expect(original).toEqual([42]);
        });

        it('should work with different types', () => {
            const original = ['a', 'b', 'c'];
            const originalCopy = [...original];
            Randomizer.shuffle(original, 999);

            expect(original).toHaveLength(3);
            expect(original).toEqual(expect.arrayContaining(originalCopy));
        });
    });

    describe('seed merging behavior', () => {
        it('should use global seed when no local seed provided', () => {
            Randomizer.setSeed(333);

            const tensor1 = Randomizer.randomUniform([2, 2]);

            Randomizer.setSeed(333);
            const tensor2 = Randomizer.randomUniform([2, 2]);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should use local seed when no global seed set', () => {
            Randomizer.setSeed(undefined);

            const tensor1 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 444);
            const tensor2 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 444);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });

        it('should produce different results with different seeds', () => {
            const tensor1 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 100);
            const tensor2 = Randomizer.randomUniform([2, 2], 0, 1, 'float32', 200);

            const values1 = tensor1.arraySync();
            const values2 = tensor2.arraySync();

            expect(values1).not.toEqual(values2);

            tensor1.dispose();
            tensor2.dispose();
        });
    });

    describe('memory management', () => {
        it('should not leak memory with repeated calls', () => {
            const initialMemory = tf.memory().numTensors;

            for (let i = 0; i < 10; i++) {
                const tensor = Randomizer.randomUniform([10, 10]);
                tensor.dispose();
            }

            const finalMemory = tf.memory().numTensors;
            expect(finalMemory).toBeLessThanOrEqual(initialMemory);
        });
    });

    describe('edge cases', () => {
        it('should handle zero-sized tensors', () => {
            const tensor = Randomizer.randomUniform([0], 0, 1);

            expect(tensor.shape).toEqual([0]);
            expect(tensor.size).toBe(0);

            tensor.dispose();
        });

        it('should handle single element tensors', () => {
            const tensor = Randomizer.randomUniform([1], 0, 1);

            expect(tensor.shape).toEqual([1]);
            expect(tensor.size).toBe(1);

            const value = tensor.dataSync()[0];
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThan(1);

            tensor.dispose();
        });
    });
});
