import { describe, it, expect } from 'vitest';
import { shuffleArray } from './shuffle';

describe('shuffleArray', () => {
    it('should contain all original elements', () => {
        const input = [1, 2, 3, 4, 5];
        const result = shuffleArray(input);

        expect(result).toEqual(expect.arrayContaining(input));
        expect(input).toEqual(expect.arrayContaining(result));
    });

    it('should work with empty array', () => {
        const input: number[] = [];
        const result = shuffleArray(input);

        expect(result).toEqual([]);
    });

    it('should work with single element array', () => {
        const input = [42];
        const result = shuffleArray(input);

        expect(result).toEqual([42]);
    });

    it('should work with string array', () => {
        const input = ['a', 'b', 'c', 'd'];
        const result = shuffleArray(input);

        expect(result).toHaveLength(4);
        expect(result).toEqual(expect.arrayContaining(['a', 'b', 'c', 'd']));
    });

    it('should work with object array', () => {
        const input = [{ id: 1 }, { id: 2 }, { id: 3 }];
        const result = shuffleArray(input);

        expect(result).toHaveLength(3);
        expect(result).toEqual(expect.arrayContaining(input));
    });

    it('should produce deterministic results with same seed', () => {
        const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        const seed = 12345;

        const result1 = shuffleArray([...input], seed);
        const result2 = shuffleArray([...input], seed);

        expect(result1).toEqual(result2);
    });

    it('should produce different results with different seeds', () => {
        const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        const result1 = shuffleArray([...input], 11111);
        const result2 = shuffleArray([...input], 22222);

        expect(result1).not.toEqual(result2);
    });

    it('should handle arrays with duplicate elements', () => {
        const input = [1, 1, 2, 2, 3, 3];
        const result = shuffleArray(input);

        expect(result.filter((x) => x === 1)).toHaveLength(2);
        expect(result.filter((x) => x === 2)).toHaveLength(2);
        expect(result.filter((x) => x === 3)).toHaveLength(2);
    });

    it('should shuffle large arrays efficiently', () => {
        const size = 1000;
        const input = Array.from({ length: size }, (_, i) => i);
        const result = shuffleArray(input);

        expect(result).toHaveLength(size);
        expect(result).toEqual(expect.arrayContaining(input));
    });

    describe('seeded randomization', () => {
        it('should handle seed value of 0', () => {
            const input = [1, 2, 3, 4, 5];
            const result = shuffleArray([...input], 0);

            expect(result).toHaveLength(5);
            expect(result).toEqual(expect.arrayContaining(input));
        });

        it('should handle negative seed values', () => {
            const input = [1, 2, 3, 4, 5];
            const result = shuffleArray([...input], -12345);

            expect(result).toHaveLength(5);
            expect(result).toEqual(expect.arrayContaining(input));
        });
    });

    describe('edge cases', () => {
        it('should handle very large numbers', () => {
            const input = [Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER, 1e10];
            const result = shuffleArray(input);

            expect(result).toEqual(expect.arrayContaining(input));
        });

        it('should handle arrays with all identical elements', () => {
            const input = Array(10).fill(42);
            const result = shuffleArray(input);

            expect(result).toHaveLength(10);
            expect(result.every((x) => x === 42)).toBe(true);
        });
    });
});
