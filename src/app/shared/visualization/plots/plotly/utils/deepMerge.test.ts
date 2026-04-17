import { describe, it, expect } from 'vitest';
import { deepMerge } from './deepMerge';

type Records = Record<string, unknown>;

describe('deepMerge security', () => {
    it('should not allow prototype pollution via __proto__', () => {
        const maliciousPayload = JSON.parse('{"__proto__": {"polluted": "yes"}}');
        const target = {};

        const result = deepMerge(target, maliciousPayload);

        // @ts-expect-error - checking prototype pollution didn't occur
        expect({}.polluted).toBeUndefined();
        expect(result).not.toHaveProperty('__proto__');
        expect(result).not.toHaveProperty('polluted');
    });

    it('should not allow prototype pollution via constructor', () => {
        const maliciousPayload = JSON.parse('{"constructor": {"prototype": {"polluted": "yes"}}}');
        const target = {};

        const result = deepMerge(target, maliciousPayload);

        // @ts-expect-error - checking prototype pollution didn't occur
        expect({}.polluted).toBeUndefined();
        expect(result).not.toHaveProperty('constructor');
    });

    it('should not allow prototype pollution via prototype key', () => {
        const maliciousPayload = JSON.parse('{"prototype": {"polluted": "yes"}}');
        const target = {};

        const result = deepMerge(target, maliciousPayload);

        // @ts-expect-error - checking prototype pollution didn't occur
        expect({}.polluted).toBeUndefined();
        expect(result).not.toHaveProperty('prototype');
    });

    it('should deep merge nested objects', () => {
        const target = { a: 1, b: { c: 2 } } as Records;
        const source = { b: { d: 3 }, e: 4 } as Records;
        const result = deepMerge(target, source);

        expect(result).toEqual({ a: 1, b: { c: 2, d: 3 }, e: 4 });
    });

    it('should overwrite non-object values', () => {
        const target = { a: 1, b: { c: 2 } } as Records;
        const source = { a: 10, b: 20 } as Records;
        const result = deepMerge(target, source);

        expect(result).toEqual({ a: 10, b: 20 });
    });

    it('should handle arrays by overwriting', () => {
        const target = { a: [1, 2] } as Records;
        const source = { a: [3, 4] } as Records;
        const result = deepMerge(target, source);

        expect(result).toEqual({ a: [3, 4] });
    });

    it('should not mutate the target object', () => {
        const target = { a: 1, b: { c: 2 } } as Records;
        const source = { b: { d: 3 } } as Records;
        deepMerge(target, source);

        expect(target).toEqual({ a: 1, b: { c: 2 } });
    });
});
