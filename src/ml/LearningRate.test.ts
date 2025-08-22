import { describe, it, expect } from 'vitest';
import { LearningRate } from './LearningRate';

describe('LearningRate', () => {
    it('should create with default parameters', () => {
        const lr = new LearningRate();
        expect(lr.next(0)).toBeCloseTo(1e-3);
    });

    it('should compute decayed learning rate', () => {
        const lr = new LearningRate(0.1, 10, 0.5);
        const rate = lr.next(5);
        expect(rate).toBeCloseTo(0.1 * Math.pow(10 / 15, 0.5));
    });

    it('should throw error for non-positive lambda', () => {
        expect(() => new LearningRate(0, 1, 0.5)).toThrow(
            'Learning rate (lambda) must be positive',
        );
    });

    it('should throw error for non-positive s0', () => {
        expect(() => new LearningRate(1e-3, -1, 0.5)).toThrow('s0 must be positive or zero');
    });

    it('should throw error for non-positive p', () => {
        expect(() => new LearningRate(1e-3, 1, -1)).toThrow('p must be positive or zero');
    });
});
