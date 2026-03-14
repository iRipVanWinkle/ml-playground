import { beforeEach, describe, expect, it } from 'vitest';
import { Randomizer } from '../../random/Randomizer';
import { IsolationSplitStrategy } from './IsolationSplitStrategy';

describe('IsolationSplitStrategy', () => {
    beforeEach(() => {
        // Reset global RNG for deterministic behaviour in tests
        Randomizer.setSeed(42);
    });

    it('returns null for empty features or insufficient indices', () => {
        const strat = new IsolationSplitStrategy(0);

        expect(strat.findBestSplit([], [])).toBeNull();
        expect(strat.findBestSplit([0], [[1, 2]])).toBeNull();
    });

    it('returns null when the selected feature has no variability', () => {
        const features = [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ];
        const indices = [0, 1, 2];

        const strat = new IsolationSplitStrategy(0);

        const res = strat.findBestSplit(indices, features);
        expect(res).toBeNull();
    });

    it('produces a valid split for variable data', () => {
        const features = [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ];
        const indices = [0, 1, 2, 3];

        // Deterministic global seed + baseSeed ensures reproducible split
        Randomizer.setSeed(123);

        const strat = new IsolationSplitStrategy(7);
        const res = strat.findBestSplit(indices, features);

        expect(res).not.toBeNull();

        if (res) {
            expect(res.featureIndex).toBeGreaterThanOrEqual(0);
            expect(res.featureIndex).toBeLessThan(features[0].length);

            const colValues = features.map((r) => r[res.featureIndex]);
            const min = Math.min(...colValues);
            const max = Math.max(...colValues);

            expect(res.threshold).toBeGreaterThanOrEqual(min);
            expect(res.threshold).toBeLessThan(max);

            expect(res.leftIndices.length + res.rightIndices.length).toBe(indices.length);
            expect(res.impurityReduction).toBe(1);
        }
    });

    it('is deterministic when using the same baseSeed and global seed', () => {
        const features = [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ];
        const indices = [0, 1, 2, 3];

        Randomizer.setSeed(999);

        const s1 = new IsolationSplitStrategy(5);
        const out1 = s1.findBestSplit(indices, features);

        Randomizer.setSeed(999);
        const s2 = new IsolationSplitStrategy(5);
        const out2 = s2.findBestSplit(indices, features);

        expect(out1).not.toBeNull();
        expect(out2).not.toBeNull();

        if (out1 && out2) {
            expect(out1.featureIndex).toBe(out2.featureIndex);
            expect(out1.threshold).toBeCloseTo(out2.threshold, 10);
            expect(out1.leftIndices).toEqual(out2.leftIndices);
            expect(out1.rightIndices).toEqual(out2.rightIndices);
        }
    });
});
