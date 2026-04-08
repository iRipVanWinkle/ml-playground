import { describe, it, expect } from 'vitest';
import { assert, assertModelTrained } from './assert';

describe('assert utils', () => {
    describe('assert', () => {
        it('should not throw if condition is true', () => {
            expect(() => assert(true, 'error message')).not.toThrow();
        });

        it('should throw an Error with the correct message if condition is false', () => {
            const message = 'custom error message';
            expect(() => assert(false, message)).toThrow(Error);
            expect(() => assert(false, message)).toThrow(message);
        });
    });

    describe('assertModelTrained', () => {
        it('should not throw for non-null values', () => {
            expect(() => assertModelTrained({})).not.toThrow();
            expect(() => assertModelTrained([1, 2])).not.toThrow();
            expect(() => assertModelTrained(0)).not.toThrow();
            expect(() => assertModelTrained('trained')).not.toThrow();
        });

        it('should throw if value is null', () => {
            expect(() => assertModelTrained(null)).toThrow('Model has not been trained yet. Please call train() first.');
        });

        it('should throw if value is undefined', () => {
            expect(() => assertModelTrained(undefined)).toThrow('Model has not been trained yet. Please call train() first.');
        });

        it('should throw if value is an empty array', () => {
            expect(() => assertModelTrained([])).toThrow('Model has not been trained yet. Please call train() first.');
        });
    });
});
