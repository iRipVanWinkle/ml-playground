import { assert } from './utils/assert';

/**
 * Learning rate scheduler that follows a decay strategy.
 */
export class LearningRate {
    private lambda: number;
    private s0: number;
    private p: number;

    constructor(lambda = 1e-3, s0 = 1, p = 0.5) {
        assert(lambda > 0, 'Learning rate (lambda) must be positive');
        assert(s0 >= 0, 's0 must be positive or zero');
        assert(p >= 0, 'p must be positive or zero');

        this.lambda = lambda;
        this.s0 = s0;
        this.p = p;
    }

    /**
     * Get the next learning rate.
     * @param iteration - The current iteration number.
     * @returns The next learning rate.
     */
    next(iteration: number): number {
        return this.lambda * Math.pow(this.s0 / (this.s0 + iteration), this.p);
    }
}
