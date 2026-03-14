import { describe, it, expect, afterAll } from 'vitest';
import { tensor2d } from '@tensorflow/tfjs';
import type { TreeNode } from '../../types';
import { expectedPathLength, IsolationForest, pathLength, quantile } from './IsolationForest';
describe('IsolationForest', () => {
    // Tree construction helpers
    function leaf(value: number): TreeNode {
        return { featureIndex: null, threshold: null, value, leftChild: null, rightChild: null };
    }

    function inner(
        featureIndex: number,
        threshold: number,
        value: number,
        leftChild: TreeNode | null,
        rightChild: TreeNode | null,
    ): TreeNode {
        return { featureIndex, threshold, value, leftChild, rightChild };
    }

    // Test dataset: 20 inliers around (0, 0) and 2 extreme outliers.
    // scikit-learn IsolationForest (n_estimators=100, max_samples=22,
    // contamination=0.1, random_state=42) labels both outliers as -1.
    const INLIERS = [
        [0.1, 0.2],
        [-0.1, 0.3],
        [0.2, -0.1],
        [-0.2, 0.1],
        [0.3, 0.0],
        [-0.3, -0.2],
        [0.0, 0.4],
        [0.1, -0.3],
        [-0.1, -0.1],
        [0.2, 0.3],
        [0.0, -0.2],
        [-0.2, 0.0],
        [0.3, 0.3],
        [-0.3, 0.1],
        [0.1, 0.1],
        [-0.1, 0.2],
        [0.2, -0.2],
        [0.0, 0.0],
        [-0.1, -0.3],
        [0.3, -0.1],
    ];

    const OUTLIERS = [
        [100.0, 100.0],
        [-100.0, -100.0],
    ];

    const ALL_POINTS = [...INLIERS, ...OUTLIERS];
    describe('expectedPathLength', () => {
        it('expected 0 for n = 0', () => {
            expect(expectedPathLength(0)).toBe(0);
        });

        it('expected 0 for n = 1', () => {
            expect(expectedPathLength(1)).toBe(0);
        });

        it('expected 1 for n = 2', () => {
            expect(expectedPathLength(2)).toBe(1);
        });

        it('expected 1.20739235758 for n = 3', () => {
            expect(expectedPathLength(3)).toBeCloseTo(1.207392357586557, 10);
        });

        it('expected 3.74888048447 for n = 10', () => {
            expect(expectedPathLength(10)).toBeCloseTo(3.74888048447244, 10);
        });

        it('expected 10.24477092011 for n = 256', () => {
            expect(expectedPathLength(256)).toBeCloseTo(10.244770920116851, 10);
        });
    });

    describe('quantile', () => {
        it('expected minimum for q = 0', () => {
            expect(quantile([1, 2, 3, 4, 5], 0)).toBeCloseTo(1, 10);
        });

        it('expected maximum for q = 1', () => {
            expect(quantile([1, 2, 3, 4, 5], 1)).toBeCloseTo(5, 10);
        });

        it('expected median for q = 0.5', () => {
            expect(quantile([1, 2, 3, 4, 5], 0.5)).toBeCloseTo(3, 10);
        });

        it('expected 0.9 quantile — linear interpolation', () => {
            expect(quantile([1, 2, 3, 4, 5], 0.9)).toBeCloseTo(4.6, 10);
        });

        it('expected 0.75 quantile for unsorted input', () => {
            expect(quantile([0.5, 0.7, 0.2, 0.9, 0.1], 0.75)).toBeCloseTo(0.7, 10);
        });

        it('single-element array returns that element for any q', () => {
            expect(quantile([42], 0)).toBe(42);
            expect(quantile([42], 0.5)).toBe(42);
            expect(quantile([42], 1)).toBe(42);
        });
    });

    describe('pathLength', () => {
        const treeA: TreeNode = leaf(10);
        const treeB: TreeNode = inner(0, 5.0, 0, leaf(3), leaf(5));
        const treeC: TreeNode = inner(1, 3.0, 0, inner(0, 2.0, 0, leaf(2), null), leaf(8));

        it('tree A (leaf, value=10) — any sample gives expectedPathLength(10)', () => {
            expect(pathLength([0, 0], treeA)).toBeCloseTo(3.74888048447244, 8);
            expect(pathLength([99, -99], treeA)).toBeCloseTo(3.74888048447244, 8);
        });

        it('tree B — sample [3.0, 0] routed to left leaf', () => {
            // feat 0: 3.0 < 5.0 → left (value=3); depth=1; 1 + epl(3) ≈ 2.207392
            expect(pathLength([3.0, 0.0], treeB)).toBeCloseTo(2.207392357586557, 8);
        });

        it('tree B — sample [7.0, 0] routed to right leaf', () => {
            // feat 0: 7.0 ≥ 5.0 → right (value=5); depth=1; 1 + epl(5) ≈ 3.327020
            expect(pathLength([7.0, 0.0], treeB)).toBeCloseTo(3.327020052039781, 8);
        });

        it('tree C — sample [1.0, 2.0] routes left→left leaf, depth 2', () => {
            // feat 1: 2.0 < 3.0 → left; feat 0: 1.0 < 2.0 → left; leaf(2)
            // depth=2; 2 + epl(2) = 2 + 1 = 3.0
            expect(pathLength([1.0, 2.0], treeC)).toBeCloseTo(3.0, 8);
        });

        it('tree C — sample [3.0, 2.0] routes left then no right-child, breaks at depth 1', () => {
            // feat 1: 2.0 < 3.0 → left inner; feat 0: 3.0 ≥ 2.0 → rightChild=null, break
            // current = inner node (value=0), depth=1; 1 + epl(0) = 1 + 0 = 1.0
            expect(pathLength([3.0, 2.0], treeC)).toBeCloseTo(1.0, 8);
        });

        it('tree C — sample [0.0, 5.0] routes to right leaf', () => {
            // feat 1: 5.0 ≥ 3.0 → right leaf (value=8); depth=1; 1 + epl(8) ≈ 4.296252
            expect(pathLength([0.0, 5.0], treeC)).toBeCloseTo(4.296251627910626, 8);
        });
    });
    describe('IsolationForest model', () => {
        const X = tensor2d(ALL_POINTS);

        afterAll(() => X.dispose());

        describe('train', () => {
            it('higher contamination produces a lower score threshold', async () => {
                const low = new IsolationForest({
                    estimators: 50,
                    maxSamples: 22,
                    contamination: 0.05,
                });
                const high = new IsolationForest({
                    estimators: 50,
                    maxSamples: 22,
                    contamination: 0.2,
                });

                const treesLow = await low.train(X);
                const treesHigh = await high.train(X);

                expect(treesLow.trees.length).toEqual(50);
                expect(treesHigh.trees.length).toEqual(50);

                expect(treesLow.scoreThreshold).toBeGreaterThan(0);
                expect(treesLow.scoreThreshold).toBeLessThan(1);
                expect(treesHigh.scoreThreshold).toBeGreaterThan(0);
                expect(treesHigh.scoreThreshold).toBeLessThan(1);

                // Higher contamination → quantile at smaller q → lower threshold
                expect(treesHigh.scoreThreshold).toBeLessThan(treesLow.scoreThreshold);

                low.dispose();
                high.dispose();
            });
        });

        describe('predict', () => {
            it('labels both extreme outliers as −1', async () => {
                // Extreme outliers at (100,100) and (-100,-100) have paths so short
                // that any reasonable random split strategy must isolate them first.
                const model = new IsolationForest({
                    estimators: 100,
                    maxSamples: 22,
                    contamination: 0.1,
                });
                await model.train(X);

                const pred = model.predict(X);
                const labels = pred.arraySync();

                expect(pred.shape).toEqual([22, 1]);

                // Indices 20 and 21 are the two outliers.
                expect(labels[20][0]).toBe(-1);
                expect(labels[21][0]).toBe(-1);

                pred.dispose();
                model.dispose();
            });
        });

        describe('predictWithMetadata', () => {
            it('anomaly score for an extreme outlier is higher than for the centroid', async () => {
                const model = new IsolationForest({
                    estimators: 100,
                    maxSamples: 22,
                    contamination: 0.1,
                });
                await model.train(X);

                const centroidTensor = tensor2d([[0.0, 0.0]]);
                const outlierTensor = tensor2d([[100.0, 100.0]]);

                const centroidMeta = model.predictWithMetadata(centroidTensor);
                const outlierMeta = model.predictWithMetadata(outlierTensor);

                const centroidScore = centroidMeta.probabilities.arraySync()[0][0];
                const outlierScore = outlierMeta.probabilities.arraySync()[0][0];

                // Anomalies have shorter paths → higher 2^(-path/c) scores.
                expect(outlierScore).toBeGreaterThan(centroidScore);

                centroidMeta.dispose();
                outlierMeta.dispose();
                centroidTensor.dispose();
                outlierTensor.dispose();
                model.dispose();
            });
        });
    });
});
