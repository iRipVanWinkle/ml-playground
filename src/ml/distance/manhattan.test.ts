import { describe, expect, it } from 'vitest';
import { tensor2d } from '@tensorflow/tfjs';
import { manhattanDistance } from './manhattan';

describe('manhattanDistance', () => {
    it('calculates zero distance for identical vectors', () => {
        const X = tensor2d([[1, 2, 3]]);
        const Y = tensor2d([[1, 2, 3]]);

        const result = manhattanDistance(X, Y);
        const distance = result.arraySync()[0][0];

        expect(distance).toBeCloseTo(0, 5);
    });

    it('calculates distance for orthogonal vectors', () => {
        const X = tensor2d([[1, 0]]);
        const Y = tensor2d([[0, 1]]);

        const result = manhattanDistance(X, Y);
        const distance = result.arraySync()[0][0];

        expect(distance).toBeCloseTo(2, 5);
    });

    it('calculates distance for opposite vectors', () => {
        const X = tensor2d([[1, 0]]);
        const Y = tensor2d([[-1, 0]]);

        const result = manhattanDistance(X, Y);
        const distance = result.arraySync()[0][0];

        expect(distance).toBeCloseTo(2, 5);
    });

    it('calculates distances for multiple vectors', () => {
        const X = tensor2d([
            [1, 0],
            [0, 1],
        ]);
        const Y = tensor2d([
            [1, 0],
            [0, 1],
        ]);

        const result = manhattanDistance(X, Y);
        const distances = result.arraySync();

        expect(distances[0][0]).toBeCloseTo(0, 5);
        expect(distances[1][1]).toBeCloseTo(0, 5);
        expect(distances[0][1]).toBeCloseTo(2, 5);
        expect(distances[1][0]).toBeCloseTo(2, 5);
    });
});
