import { describe, it, expect } from 'vitest';
import { tensor2d, tensor1d } from '@tensorflow/tfjs';
import { silhouetteSample, silhouetteCluster, silhouetteScore } from './silhouette';
import { manhattanDistance } from '../../distance';

describe('silhouette', () => {
    describe('silhouetteSample', () => {
        it('should calculate silhouette scores for well-separated clusters', () => {
            const X = tensor2d([
                [0, 0],
                [1, 0],
                [10, 0],
                [11, 0],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = silhouetteSample(X, labels, numClusters);
            const resultArray = Array.from(result.dataSync());

            resultArray.forEach((score) => {
                expect(score).toBeGreaterThan(0.5);
                expect(score).toBeLessThanOrEqual(1);
            });

            result.dispose();
            X.dispose();
            labels.dispose();
        });

        it('should work with manhattan distance', () => {
            const X = tensor2d([
                [0, 0],
                [1, 1],
                [10, 10],
                [11, 11],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = silhouetteSample(X, labels, numClusters, manhattanDistance);
            const resultArray = Array.from(result.dataSync());

            resultArray.forEach((score) => {
                expect(score).toBeGreaterThan(0);
                expect(score).toBeLessThanOrEqual(1);
            });

            result.dispose();
            X.dispose();
            labels.dispose();
        });
    });

    describe('silhouetteCluster', () => {
        it('should calculate average silhouette scores per cluster', () => {
            const X = tensor2d([
                [0, 0],
                [1, 0],
                [10, 0],
                [11, 0],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = silhouetteCluster(X, labels, numClusters);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray).toHaveLength(2);
            expect(resultArray[0]).toBeGreaterThan(0.5);
            expect(resultArray[1]).toBeGreaterThan(0.5);

            result.dispose();
            X.dispose();
            labels.dispose();
        });

        it('should work with pre-calculated sample scores', () => {
            const sampleScores = tensor1d([0.8, 0.9, 0.7, 0.6]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = silhouetteCluster(sampleScores, labels, numClusters);
            const resultArray = Array.from(result.dataSync());

            expect(resultArray[0]).toBeCloseTo(0.85, 5);
            expect(resultArray[1]).toBeCloseTo(0.65, 5);

            result.dispose();
            sampleScores.dispose();
            labels.dispose();
        });
    });

    describe('silhouetteScore', () => {
        it('should calculate overall silhouette score', () => {
            const X = tensor2d([
                [0, 0],
                [1, 0],
                [10, 0],
                [11, 0],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]]);
            const numClusters = 2;

            const result = silhouetteScore(X, labels, numClusters);
            const score = result.dataSync()[0];

            expect(score).toBeGreaterThan(0.5);
            expect(score).toBeLessThanOrEqual(1);

            result.dispose();
            X.dispose();
            labels.dispose();
        });

        it('should work with pre-calculated sample scores', () => {
            const sampleScores = tensor1d([0.8, 0.9, 0.7, 0.6]);

            const result = silhouetteScore(sampleScores);
            const score = result.dataSync()[0];

            expect(score).toBeCloseTo(0.75, 5);

            result.dispose();
            sampleScores.dispose();
        });

        it('should work with manhattan distance', () => {
            const X = tensor2d([
                [0, 0],
                [1, 1],
                [10, 10],
                [11, 11],
            ]);
            const labels = tensor2d([[0], [0], [1], [1]], [4, 1], 'int32');
            const numClusters = 2;

            const result = silhouetteScore(X, labels, numClusters, manhattanDistance);
            const score = result.dataSync()[0];

            expect(score).toBeGreaterThan(0.5);
            expect(score).toBeLessThanOrEqual(1);

            result.dispose();
            X.dispose();
            labels.dispose();
        });
    });
});
