import { describe, it, expect } from 'vitest';
import { tensor2d } from '@tensorflow/tfjs';
import { distanceToOtherCenters, avgDistanceToOtherCenters } from './separation';
import { manhattanDistance } from '../../distance';

describe('separation', () => {
    describe('distanceToOtherCenters', () => {
        it('should work with multiple clusters', () => {
            const X = tensor2d([
                [0, 0],
                [5, 0],
                [10, 0],
            ]);
            const labels = tensor2d([[0], [1], [2]]);
            const centers = tensor2d([
                [0, 0],
                [5, 0],
                [10, 0],
            ]);

            const result = distanceToOtherCenters(X, labels, centers);
            const resultArray = result.arraySync();

            expect(resultArray[0][0]).toBeCloseTo(0, 5);
            expect(resultArray[0][1]).toBeCloseTo(5, 5);
            expect(resultArray[0][2]).toBeCloseTo(10, 5);

            expect(resultArray[1][0]).toBeCloseTo(5, 5);
            expect(resultArray[1][1]).toBeCloseTo(0, 5);
            expect(resultArray[1][2]).toBeCloseTo(5, 5);

            expect(resultArray[2][0]).toBeCloseTo(10, 5);
            expect(resultArray[2][1]).toBeCloseTo(5, 5);
            expect(resultArray[2][2]).toBeCloseTo(0, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });

        it('should work with manhattan distance', () => {
            const X = tensor2d([
                [0, 0],
                [3, 4],
            ]);
            const labels = tensor2d([[0], [1]]);
            const centers = tensor2d([
                [0, 0],
                [3, 4],
            ]);

            const result = distanceToOtherCenters(X, labels, centers, manhattanDistance);
            const resultArray = result.arraySync();

            expect(resultArray[0][0]).toBeCloseTo(0, 5);
            expect(resultArray[0][1]).toBeCloseTo(7, 5);
            expect(resultArray[1][0]).toBeCloseTo(7, 5);
            expect(resultArray[1][1]).toBeCloseTo(0, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });
    });

    describe('avgDistanceToOtherCenters', () => {
        it('should calculate average distances to other centers per cluster', () => {
            const X = tensor2d([
                [0, 0],
                [5, 0],
                [10, 0],
                [15, 0],
            ]);
            const labels = tensor2d([[0], [1], [2], [3]]);
            const centers = tensor2d([
                [0, 0],
                [5, 0],
                [10, 0],
                [15, 0],
            ]);

            const result = avgDistanceToOtherCenters(X, labels, centers);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(10, 5);
            expect(resultArray[1]).toBeCloseTo(6.667, 2);
            expect(resultArray[2]).toBeCloseTo(6.667, 2);
            expect(resultArray[3]).toBeCloseTo(10, 5);

            result.dispose();
            X.dispose();
            labels.dispose();
            centers.dispose();
        });

        it('should work with pre-calculated distances', () => {
            const distances = tensor2d([
                [0, 5, 10],
                [5, 0, 5],
                [10, 5, 0],
            ]);
            const labels = tensor2d([[0], [1], [2]]);
            const numClusters = 3;

            const result = avgDistanceToOtherCenters(distances, labels, numClusters);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(7.5, 5);
            expect(resultArray[1]).toBeCloseTo(5, 5);
            expect(resultArray[2]).toBeCloseTo(7.5, 5);

            result.dispose();
            distances.dispose();
            labels.dispose();
        });
    });
});
