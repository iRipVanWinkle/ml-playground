import { describe, it, expect } from 'vitest';
import { tensor2d } from '@tensorflow/tfjs';
import { AgglomerativeClustering } from './AgglomerativeClustering';
import { EventEmitter } from '../../events/EventEmitter';
import { manhattanDistance } from '../../distance';

describe('AgglomerativeClustering', () => {
    // Three well-separated 2-D clusters (4 points each)
    const X = tensor2d([
        // Cluster A – around (1, 1)
        [1.0, 1.0],
        [1.1, 1.0],
        [1.0, 1.1],
        [1.1, 1.1],
        // Cluster B – around (5, 5)
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],
        [5.1, 5.1],
        // Cluster C – around (9, 9)
        [9.0, 9.0],
        [9.1, 9.0],
        [9.0, 9.1],
        [9.1, 9.1],
    ]);

    describe('ward linkage (default)', () => {
        it('should group all points into the correct 3 clusters', async () => {
            const model = new AgglomerativeClustering({ numClusters: 3 });
            await model.train(X);

            const preds = model.predict(X);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            model.dispose();

            // Each group of 4 must share the same label
            const labelA = labels[0];
            const labelB = labels[4];
            const labelC = labels[8];

            // All three labels must be distinct
            expect(new Set([labelA, labelB, labelC]).size).toBe(3);

            // Within-group consistency
            expect(labels.slice(0, 4).every((l) => l === labelA)).toBe(true);
            expect(labels.slice(4, 8).every((l) => l === labelB)).toBe(true);
            expect(labels.slice(8, 12).every((l) => l === labelC)).toBe(true);
        });

        it('should return correct params shape', async () => {
            const model = new AgglomerativeClustering({ numClusters: 3 });
            const params = await model.train(X);

            expect(params.centroids.shape[0]).toBe(3);
            expect(params.centroids.shape[1]).toBe(2);
            expect(params.assignments.length).toBe(12);

            model.dispose();
        });
    });

    describe('single linkage', () => {
        it('should cluster well-separated data correctly', async () => {
            const model = new AgglomerativeClustering({
                numClusters: 3,
                linkage: 'single',
            });
            await model.train(X);

            const preds = model.predict(X);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            model.dispose();

            const labelA = labels[0];
            const labelB = labels[4];
            const labelC = labels[8];

            expect(new Set([labelA, labelB, labelC]).size).toBe(3);
            expect(labels.slice(0, 4).every((l) => l === labelA)).toBe(true);
            expect(labels.slice(4, 8).every((l) => l === labelB)).toBe(true);
            expect(labels.slice(8, 12).every((l) => l === labelC)).toBe(true);
        });
    });

    describe('complete linkage', () => {
        it('should cluster well-separated data correctly', async () => {
            const model = new AgglomerativeClustering({
                numClusters: 3,
                linkage: 'complete',
            });
            await model.train(X);

            const preds = model.predict(X);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            model.dispose();

            const labelA = labels[0];
            const labelB = labels[4];
            const labelC = labels[8];

            expect(new Set([labelA, labelB, labelC]).size).toBe(3);
            expect(labels.slice(0, 4).every((l) => l === labelA)).toBe(true);
            expect(labels.slice(4, 8).every((l) => l === labelB)).toBe(true);
            expect(labels.slice(8, 12).every((l) => l === labelC)).toBe(true);
        });
    });

    describe('average linkage', () => {
        it('should cluster well-separated data correctly', async () => {
            const model = new AgglomerativeClustering({
                numClusters: 3,
                linkage: 'average',
            });
            await model.train(X);

            const preds = model.predict(X);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            model.dispose();

            const labelA = labels[0];
            const labelB = labels[4];
            const labelC = labels[8];

            expect(new Set([labelA, labelB, labelC]).size).toBe(3);
            expect(labels.slice(0, 4).every((l) => l === labelA)).toBe(true);
            expect(labels.slice(4, 8).every((l) => l === labelB)).toBe(true);
            expect(labels.slice(8, 12).every((l) => l === labelC)).toBe(true);
        });
    });

    describe('predictWithMetadata', () => {
        it('should return ClusteringMetadata with correct type', async () => {
            const model = new AgglomerativeClustering({ numClusters: 3 });
            await model.train(X);

            const meta = model.predictWithMetadata(X);

            expect(meta.type).toBe('clustering');
            expect(meta.assignments.shape).toEqual([12, 1]);

            meta.dispose();
            model.dispose();
        });
    });

    describe('event emitter callbacks', () => {
        it('should emit n - numClusters merge callbacks', async () => {
            const emitter = new EventEmitter();
            const iterations: number[] = [];

            emitter.on('callback', (params) => {
                iterations.push(params.iteration);
            });

            const model = new AgglomerativeClustering({
                numClusters: 3,
                eventEmitter: emitter,
            });
            await model.train(X);

            // 12 points → 3 clusters requires exactly 9 merges
            expect(iterations.length).toBe(9);
            expect(iterations[0]).toBe(0);
            expect(iterations[8]).toBe(8);

            model.dispose();
        });

        it('should report decreasing numClusters in each callback', async () => {
            const emitter = new EventEmitter();
            const clusterCounts: number[] = [];

            emitter.on('callback', (params) => {
                if ('numClusters' in params) {
                    clusterCounts.push(params.numClusters);
                }
            });

            const model = new AgglomerativeClustering({
                numClusters: 3,
                eventEmitter: emitter,
            });
            await model.train(X);

            // numClusters should reduce by 1 each step: 11, 10, ..., 3
            expect(clusterCounts[0]).toBe(11);
            expect(clusterCounts[clusterCounts.length - 1]).toBe(3);

            for (let i = 1; i < clusterCounts.length; i++) {
                expect(clusterCounts[i]).toBe(clusterCounts[i - 1] - 1);
            }

            model.dispose();
        });
    });

    describe('validation', () => {
        it('should throw when numClusters < 2', () => {
            expect(() => new AgglomerativeClustering({ numClusters: 1 })).toThrow(
                'Number of clusters must be at least 2.',
            );
        });

        it('should throw when numClusters > number of samples', async () => {
            const small = tensor2d([
                [1, 2],
                [3, 4],
            ]);
            const model = new AgglomerativeClustering({ numClusters: 5 });
            await expect(model.train(small)).rejects.toThrow();
            small.dispose();
            model.dispose();
        });

        it('should throw on predict before train', () => {
            const model = new AgglomerativeClustering({ numClusters: 2 });
            expect(() => model.predict(X)).toThrow('Model has not been trained yet');
            model.dispose();
        });

        it('should throw when linkage is ward and distance is non-Euclidean', () => {
            expect(
                () =>
                    new AgglomerativeClustering({
                        numClusters: 3,
                        linkage: 'ward',
                        distanceMetric: manhattanDistance,
                    }),
            ).toThrow("Ward's linkage method requires the Euclidean distance metric.");
        });
    });

    describe('predict with explicit params', () => {
        it('should accept externally supplied params', async () => {
            const model = new AgglomerativeClustering({ numClusters: 3 });
            const params = await model.train(X);

            const freshModel = new AgglomerativeClustering({ numClusters: 3 });
            const preds = freshModel.predict(X, params);
            const labels = Array.from(preds.dataSync());

            expect(labels).toHaveLength(12);
            expect(new Set(labels).size).toBe(3);

            preds.dispose();
            model.dispose();
            freshModel.dispose();
        });
    });
});
