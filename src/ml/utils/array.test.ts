import { describe, it, expect } from 'vitest';
import { range, zeros, gather } from './array';

describe('array utils', () => {
    describe('range', () => {
        it('should create an array of numbers from 0 to n-1', () => {
            expect(range(5)).toEqual([0, 1, 2, 3, 4]);
        });

        it('should handle n = 0', () => {
            expect(range(0)).toEqual([]);
        });

        it('should handle n = 1', () => {
            expect(range(1)).toEqual([0]);
        });
    });

    describe('zeros', () => {
        it('should create a 1D array filled with zeros', () => {
            expect(zeros([5])).toEqual([0, 0, 0, 0, 0]);
        });

        it('should create a 2D array filled with zeros', () => {
            expect(zeros([2, 3])).toEqual([
                [0, 0, 0],
                [0, 0, 0],
            ]);
        });

        it('should handle 0 elements in 1D array', () => {
            expect(zeros([0])).toEqual([]);
        });

        it('should handle 0 elements in 2D array rows', () => {
            expect(zeros([0, 3])).toEqual([]);
        });

        it('should handle 0 elements in 2D array cols', () => {
            expect(zeros([3, 0])).toEqual([[], [], []]);
        });
    });

    describe('gather', () => {
        describe('1D array', () => {
            it('should gather elements based on provided indices', () => {
                const features = [10, 20, 30, 40, 50];
                const indices = [1, 3, 4];
                expect(gather(features, indices)).toEqual([20, 40, 50]);
            });

            it('should handle empty indices', () => {
                const features = [10, 20, 30];
                const indices: number[] = [];
                expect(gather(features, indices)).toEqual([]);
            });

            it('should gather elements with out of order indices', () => {
                const features = [10, 20, 30, 40];
                const indices = [3, 0, 2];
                expect(gather(features, indices)).toEqual([40, 10, 30]);
            });

            it('should handle repeating indices', () => {
                const features = [10, 20, 30];
                const indices = [1, 1, 0, 1];
                expect(gather(features, indices)).toEqual([20, 20, 10, 20]);
            });

            it('should return undefined for out of bounds indices', () => {
                const features = [10, 20];
                const indices = [0, 5];
                expect(gather(features, indices)).toEqual([10, undefined]);
            });
        });

        describe('2D array', () => {
            it('should gather rows based on provided indices', () => {
                const features = [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                ];
                const indices = [1, 3];
                expect(gather(features, indices)).toEqual([
                    [3, 4],
                    [7, 8],
                ]);
            });

            it('should handle empty indices', () => {
                const features = [
                    [1, 2],
                    [3, 4],
                ];
                const indices: number[] = [];
                expect(gather(features, indices)).toEqual([]);
            });

            it('should gather rows with out of order indices', () => {
                const features = [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ];
                const indices = [2, 0];
                expect(gather(features, indices)).toEqual([
                    [5, 6],
                    [1, 2],
                ]);
            });

            it('should handle repeating indices', () => {
                 const features = [
                    [1, 2],
                    [3, 4],
                ];
                const indices = [1, 1, 0];
                expect(gather(features, indices)).toEqual([
                    [3, 4],
                    [3, 4],
                    [1, 2],
                ]);
            });

            it('should return undefined for out of bounds indices', () => {
                const features = [
                    [1, 2],
                    [3, 4],
                ];
                const indices = [0, 5];
                expect(gather(features, indices)).toEqual([
                    [1, 2],
                    undefined
                ]);
            });
        });
    });
});
