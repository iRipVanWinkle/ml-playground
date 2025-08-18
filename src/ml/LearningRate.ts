/**
 * Learning rate scheduler that follows a decay strategy.
 */
export class LearningRate {
    private lambda: number;
    private s0: number;
    private p: number;

    constructor(lambda = 1e-3, s0 = 1, p = 0.5) {
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
