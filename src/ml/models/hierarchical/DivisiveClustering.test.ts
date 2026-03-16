import { describe, it, expect } from 'vitest';
import { tensor2d } from '@tensorflow/tfjs';
import { DivisiveClustering } from './DivisiveClustering';
import { EventEmitter } from '../../events/EventEmitter';

describe('DivisiveClustering', () => {
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

    describe('basic clustering', () => {
        it('should group all points into 3 correct clusters', async () => {
            const model = new DivisiveClustering({ numClusters: 3 });
            await model.train(X);

            const preds = model.predict(X);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            model.dispose();

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
            const model = new DivisiveClustering({ numClusters: 3 });
            const params = await model.train(X);

            expect(params.centroids.shape[0]).toBe(3);
            expect(params.centroids.shape[1]).toBe(2);
            expect(params.assignments.length).toBe(12);

            model.dispose();
        });

        it('should assign all labels in range [0, numClusters-1]', async () => {
            const model = new DivisiveClustering({ numClusters: 3 });
            await model.train(X);

            const preds = model.predict(X);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            model.dispose();

            expect(labels.every((l) => l >= 0 && l <= 2)).toBe(true);
            expect(new Set(labels).size).toBe(3);
        });
    });

    describe('predictWithMetadata', () => {
        it('should return ClusteringMetadata with correct type', async () => {
            const model = new DivisiveClustering({ numClusters: 3 });
            await model.train(X);

            const meta = model.predictWithMetadata(X);

            expect(meta.type).toBe('clustering');
            expect(meta.assignments.shape).toEqual([12, 1]);

            meta.dispose();
            model.dispose();
        });
    });

    describe('event emitter callbacks', () => {
        it('should emit numClusters - 1 split callbacks plus initial callback', async () => {
            const emitter = new EventEmitter();
            const iterations: number[] = [];

            emitter.on('callback', (params) => {
                iterations.push(params.iteration);
            });

            const model = new DivisiveClustering({
                numClusters: 3,
                eventEmitter: emitter,
            });
            await model.train(X);

            // 1 initial + (3 - 1) = 3 split callbacks total
            expect(iterations.length).toBe(3);

            model.dispose();
        });

        it('should report increasing numClusters in split callbacks', async () => {
            const emitter = new EventEmitter();
            const clusterCounts: number[] = [];

            emitter.on('callback', (params) => {
                if ('numClusters' in params) {
                    clusterCounts.push(params.numClusters);
                }
            });

            const model = new DivisiveClustering({
                numClusters: 3,
                eventEmitter: emitter,
            });
            await model.train(X);

            // First callback is the initial state (1 cluster)
            expect(clusterCounts[0]).toBe(1);
            // Last callback reaches 3 clusters
            expect(clusterCounts[clusterCounts.length - 1]).toBe(3);

            // Each subsequent count must be one greater
            for (let i = 1; i < clusterCounts.length; i++) {
                expect(clusterCounts[i]).toBe(clusterCounts[i - 1] + 1);
            }

            model.dispose();
        });
    });

    describe('validation', () => {
        it('should throw when numClusters < 2', () => {
            expect(() => new DivisiveClustering({ numClusters: 1 })).toThrow(
                'Number of clusters must be at least 2.',
            );
        });

        it('should throw when numClusters > number of samples', async () => {
            const small = tensor2d([
                [1, 2],
                [3, 4],
            ]);
            const model = new DivisiveClustering({ numClusters: 5 });
            await expect(model.train(small)).rejects.toThrow();
            small.dispose();
            model.dispose();
        });

        it('should throw on predict before train', () => {
            const model = new DivisiveClustering({ numClusters: 2 });
            expect(() => model.predict(X)).toThrow('Model has not been trained yet');
            model.dispose();
        });
    });

    describe('predict with explicit params', () => {
        it('should accept externally supplied params', async () => {
            const model = new DivisiveClustering({ numClusters: 3 });
            const params = await model.train(X);

            const freshModel = new DivisiveClustering({ numClusters: 3 });
            const preds = freshModel.predict(X, params);
            const labels = Array.from(preds.dataSync());

            expect(labels).toHaveLength(12);
            expect(new Set(labels).size).toBe(3);

            preds.dispose();
            model.dispose();
            freshModel.dispose();
        });
    });

    describe('two clusters', () => {
        it('should correctly bisect two well-separated groups', async () => {
            const data = tensor2d([
                [0, 0],
                [0.1, 0.1],
                [10, 10],
                [10.1, 10.1],
            ]);
            const model = new DivisiveClustering({ numClusters: 2 });
            await model.train(data);

            const preds = model.predict(data);
            const labels = Array.from(preds.dataSync());
            preds.dispose();
            data.dispose();
            model.dispose();

            expect(labels[0]).toBe(labels[1]);
            expect(labels[2]).toBe(labels[3]);
            expect(labels[0]).not.toBe(labels[2]);
        });
    });
});
