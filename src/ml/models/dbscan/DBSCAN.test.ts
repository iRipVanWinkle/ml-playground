import { describe, it, expect, afterAll } from 'vitest';
import { tensor2d, memory } from '@tensorflow/tfjs';
import { DBSCAN } from './DBSCAN';
import { EventEmitter } from '../../events/EventEmitter';

describe('DBSCAN', () => {
    // Three well-separated clusters + two noise points
    const X = tensor2d([
        // Cluster 0
        [1.0, 1.0],
        [1.1, 1.0],
        [1.0, 1.1],
        [1.1, 1.1],
        // Cluster 1
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],
        [5.1, 5.1],
        // Cluster 2
        [9.0, 9.0],
        [9.1, 9.0],
        [9.0, 9.1],
        [9.1, 9.1],
        // Noise
        [50.0, 50.0],
        [100.0, 100.0],
    ]);

    afterAll(() => X.dispose());

    it('should discover correct number of clusters and noise points', async () => {
        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 3,
        });

        const params = await dbscan.train(X);

        // scikit-learn reference: all 12 non-noise points are core points
        expect(params.type).toBe('dbscan');
        expect(params.corePoints.shape[0]).toBe(12);
        expect(params.corePoints.shape[1]).toBe(2);

        dbscan.dispose();
    });

    it('should assign all points in a dense group to the same cluster', async () => {
        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 3,
        });

        await dbscan.train(X);

        // Predict on the training data
        const predictions = dbscan.predict(X);
        const labels = predictions.dataSync();

        expect(labels[0]).toBe(0);
        expect(labels[1]).toBe(0);
        expect(labels[2]).toBe(0);
        expect(labels[3]).toBe(0);

        expect(labels[4]).toBe(1);
        expect(labels[5]).toBe(1);
        expect(labels[6]).toBe(1);
        expect(labels[7]).toBe(1);

        expect(labels[8]).toBe(2);
        expect(labels[9]).toBe(2);
        expect(labels[10]).toBe(2);
        expect(labels[11]).toBe(2);

        // Noise points
        expect(labels[12]).toBe(-1);
        expect(labels[13]).toBe(-1);

        predictions.dispose();
        dbscan.dispose();
    });

    it('should mark all points as noise when epsilon is too small', async () => {
        const dbscan = new DBSCAN({
            epsilon: 0.01,
            minPoints: 3,
        });

        const params = await dbscan.train(X);

        // scikit-learn reference: 0 core points, all 14 points are noise
        expect(params.corePoints.shape[0]).toBe(0);

        const predictions = dbscan.predict(X);
        const labels = predictions.dataSync();
        expect(labels).toEqual(new Float32Array(14).fill(-1));

        predictions.dispose();
        dbscan.dispose();
    });

    it('should put all points into one cluster when epsilon is very large', async () => {
        const dbscan = new DBSCAN({
            epsilon: 200,
            minPoints: 2,
        });

        await dbscan.train(X);

        const predictions = dbscan.predict(X);
        const labels = predictions.arraySync().flat();

        // scikit-learn reference: all 14 points in cluster 0
        expect(labels).toEqual(new Array(14).fill(0));

        predictions.dispose();
        dbscan.dispose();
    });

    it('should predict new points correctly', async () => {
        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 3,
        });

        await dbscan.train(X);

        const newPoints = tensor2d([
            [1.05, 1.05], // near cluster 0
            [5.05, 5.05], // near cluster 1
            [999.0, 999.0], // far from all clusters → noise
        ]);

        const predictions = dbscan.predict(newPoints);
        const labels = predictions.arraySync().flat();

        // scikit-learn reference: predicted = [0, 1, -1]
        expect(labels[0]).toBe(0);
        expect(labels[1]).toBe(1);
        expect(labels[2]).toBe(-1);

        predictions.dispose();
        newPoints.dispose();
        dbscan.dispose();
    });

    it('should work with predictWithMetadata', async () => {
        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 3,
        });

        await dbscan.train(X);

        const result = dbscan.predictWithMetadata(X);

        expect(result.type).toBe('clustering');
        expect(result.assignments.shape).toEqual([14, 1]);

        result.dispose();
        dbscan.dispose();
    });

    it('should emit callback events for each point processed', async () => {
        const eventEmitter = new EventEmitter();
        const callbacks: {
            iteration: number;
            numClusters: number;
            activePointIndex?: number;
        }[] = [];

        eventEmitter.on(
            'callback',
            (data: {
                iteration: number;
                numClusters: number;

                activePointIndex?: number;
            }) => {
                callbacks.push({
                    iteration: data.iteration,
                    numClusters: data.numClusters,
                    activePointIndex: data.activePointIndex,
                });
            },
        );

        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 3,
            eventEmitter,
        });

        await dbscan.train(X);

        // Per-point callbacks: more than one per cluster
        expect(callbacks.length).toBeGreaterThan(3);

        // First callback should be the seed of cluster 0
        expect(callbacks[0].numClusters).toBe(1);
        expect(callbacks[0].activePointIndex).toBe(0);

        // Final callback has no active point and correct final counts
        const last = callbacks[callbacks.length - 1];
        expect(last.numClusters).toBe(3);
        expect(last.activePointIndex).toBeUndefined();

        dbscan.dispose();
    });

    it('should throw on invalid epsilon', () => {
        expect(() => new DBSCAN({ epsilon: 0, minPoints: 3 })).toThrow();
        expect(() => new DBSCAN({ epsilon: -1, minPoints: 3 })).toThrow();
    });

    it('should throw on invalid minPoints', () => {
        expect(() => new DBSCAN({ epsilon: 1, minPoints: 0 })).toThrow();
        expect(() => new DBSCAN({ epsilon: 1, minPoints: 2.5 })).toThrow();
    });

    it('should throw when predicting without training', () => {
        const dbscan = new DBSCAN({ epsilon: 1, minPoints: 3 });
        const input = tensor2d([[1, 2]]);

        expect(() => dbscan.predict(input)).toThrow('Model has not been trained');

        input.dispose();
        dbscan.dispose();
    });

    it('should work with higher-dimensional data', async () => {
        const highDimData = tensor2d([
            [1, 1, 1, 1],
            [1.1, 1, 1, 1.1],
            [1, 1.1, 1.1, 1],
            [10, 10, 10, 10],
            [10.1, 10, 10, 10.1],
            [10, 10.1, 10.1, 10],
        ]);

        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 2,
        });

        const params = await dbscan.train(highDimData);

        // scikit-learn reference: labels = [0, 0, 0, 1, 1, 1], n_clusters = 2
        expect(params.corePoints.shape[1]).toBe(4);
        expect(params.corePoints.shape[0]).toBe(6); // all points are core points

        const predictions = dbscan.predict(highDimData);
        const labels = predictions.dataSync();
        expect(labels).toEqual(new Float32Array([0, 0, 0, 1, 1, 1]));

        predictions.dispose();
        highDimData.dispose();
        dbscan.dispose();
    });

    it('should not leak memory during training and prediction', async () => {
        const dbscan = new DBSCAN({
            epsilon: 0.5,
            minPoints: 3,
        });

        const initialNumTensors = memory().numTensors;

        await dbscan.train(X);
        const predictions = dbscan.predict(X);

        predictions.dispose();
        dbscan.dispose();

        const finalNumTensors = memory().numTensors;

        expect(finalNumTensors).toBeLessThanOrEqual(initialNumTensors);
    });
});
