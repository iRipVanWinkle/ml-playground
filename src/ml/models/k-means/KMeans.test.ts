import { describe, it, expect } from 'vitest';
import { tensor2d, memory } from '@tensorflow/tfjs';
import { KMeans } from './KMeans';
import { centroidInitializationFactory } from '../../factories';
import { EventEmitter } from '../../events/EventEmitter';

describe('KMeans', () => {
    const X = tensor2d([
        [2.1, 1.5, 3.2],
        [2.3, 1.8, 3.0],
        [2.0, 1.6, 3.1],
        [5.5, 4.2, 6.1],
        [5.8, 4.5, 6.3],
        [5.3, 4.0, 5.9],
        [8.2, 7.1, 8.5],
        [8.5, 7.4, 8.8],
        [8.1, 7.0, 8.3],
        [2.2, 1.7, 3.3],
        [5.6, 4.3, 6.0],
        [8.3, 7.2, 8.6],
    ]);

    const customCentroids = tensor2d([
        [6.9, 5.7, 7.3],
        [4.3, 3.5, 4.9],
        [2.0, 0.6, 2.1],
    ]);

    const initializeCentroids = centroidInitializationFactory({
        type: 'custom',
        centroids: customCentroids,
    });

    it('should cluster data with custom initial centroids', async () => {
        const eventEmitter = new EventEmitter();

        let lastIteration = 0;
        eventEmitter.on('callback', (data) => {
            lastIteration = data.iteration;
        });

        const kmeans = new KMeans({
            numClusters: 3,
            maxIterations: 15,
            initializeCentroids,
            eventEmitter,
        });

        const finalCentroids = await kmeans.train(X);

        expect(lastIteration).toBe(14); // start from 0

        const centroidsData = await finalCentroids.array();

        expect(centroidsData[0][0]).toBeCloseTo(8.27, 2);
        expect(centroidsData[0][1]).toBeCloseTo(7.18, 2);
        expect(centroidsData[0][2]).toBeCloseTo(8.55, 2);

        expect(centroidsData[1][0]).toBeCloseTo(5.55, 2);
        expect(centroidsData[1][1]).toBeCloseTo(4.25, 2);
        expect(centroidsData[1][2]).toBeCloseTo(6.07, 2);

        expect(centroidsData[2][0]).toBeCloseTo(2.15, 2);
        expect(centroidsData[2][1]).toBeCloseTo(1.65, 2);
        expect(centroidsData[2][2]).toBeCloseTo(3.15, 2);

        kmeans.dispose();
    });

    it('should stop earlier when converged', async () => {
        const eventEmitter = new EventEmitter();

        let lastIteration = 0;
        eventEmitter.on('callback', (data) => {
            lastIteration = data.iteration;
        });

        const kmeans = new KMeans({
            numClusters: 3,
            maxIterations: 100,
            tolerance: 1e-4,
            initializeCentroids,
            eventEmitter,
        });

        await kmeans.train(X);

        expect(lastIteration).toBe(2);

        kmeans.dispose();
    });

    it('should not leak memory during training', async () => {
        const eventEmitter = new EventEmitter();

        const kmeans = new KMeans({
            numClusters: 3,
            maxIterations: 5,
            initializeCentroids,
            eventEmitter,
        });

        const initialNumTensors = memory().numTensors;

        await kmeans.train(X);

        kmeans.dispose();

        const finalNumTensors = memory().numTensors;

        expect(finalNumTensors).toBeLessThanOrEqual(initialNumTensors);
    });
});
