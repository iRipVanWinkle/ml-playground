import { describe, it, expect } from 'vitest';
import { tensor2d, tensor1d } from '@tensorflow/tfjs';
import { distanceToCenter, avgDistanceToCenter, maxDistanceToCenter } from './compactness';
import { manhattanDistance } from '../../distance';

describe('compactness', () => {
    describe('distanceToCenter', () => {
        it('should calculate correct distances for simple case', () => {
            const X = tensor2d([
                [0, 0],
                [1, 0],
                [10, 0],
                [11, 0],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]], [4, 1], 'int32');
            const centers = tensor2d([
                [0.5, 0],
                [10.5, 0],
            ]);

            const result = distanceToCenter(X, labels, centers);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(0.5, 5);
            expect(resultArray[1]).toBeCloseTo(0.5, 5);
            expect(resultArray[2]).toBeCloseTo(0.5, 5);
            expect(resultArray[3]).toBeCloseTo(0.5, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });

        it('should work with manhattan distance', () => {
            const X = tensor2d([
                [0, 0],
                [1, 1],
                [10, 10],
            ]);
            const labels = tensor2d([[0], [0], [1]], [3, 1], 'int32');
            const centers = tensor2d([
                [0, 0],
                [10, 10],
            ]);

            const result = distanceToCenter(X, labels, centers, manhattanDistance);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(0, 5);
            expect(resultArray[1]).toBeCloseTo(2, 5);
            expect(resultArray[2]).toBeCloseTo(0, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });

        it('should handle higher dimensional data', () => {
            const X = tensor2d([
                [1, 2, 3],
                [4, 5, 6],
            ]);
            const labels = tensor2d([[0], [1]]);
            const centers = tensor2d([
                [1, 2, 3],
                [4, 5, 6],
            ]);

            const result = distanceToCenter(X, labels, centers);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(0, 5);
            expect(resultArray[1]).toBeCloseTo(0, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });
    });

    describe('avgDistanceToCenter', () => {
        it('should calculate correct average distances per cluster', () => {
            const X = tensor2d([
                [0, 0],
                [2, 0],
                [10, 0],
                [12, 0],
            ]);
            const labels = tensor2d([[0], [0], [0], [1]]);
            const centers = tensor2d([
                [1, 0],
                [11, 0],
            ]);

            const result = avgDistanceToCenter(X, labels, centers);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(3.66667, 5);
            expect(resultArray[1]).toBeCloseTo(1, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });

        it('should work with pre-calculated distances', () => {
            const distances = tensor1d([1, 2, 3, 4]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = avgDistanceToCenter(distances, labels, numClusters);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(1.5, 5);
            expect(resultArray[1]).toBeCloseTo(3.5, 5);

            result.dispose();
            distances.dispose();
            labels.dispose();
        });
    });

    describe('maxDistanceToCenter', () => {
        it('should calculate correct max distances per cluster', () => {
            const X = tensor2d([
                [0, 0],
                [2, 0],
                [10, 0],
                [13, 0],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const centers = tensor2d([
                [1, 0],
                [11, 0],
            ]);

            const result = maxDistanceToCenter(X, labels, centers);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(1, 5);
            expect(resultArray[1]).toBeCloseTo(2, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });

        it('should work with pre-calculated distances', () => {
            const distances = tensor1d([1, 2, 3, 4]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = maxDistanceToCenter(distances, labels, numClusters);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(2, 5);
            expect(resultArray[1]).toBeCloseTo(4, 5);

            result.dispose();
            distances.dispose();
            labels.dispose();
        });
    });
});
