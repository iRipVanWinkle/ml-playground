import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { CriterionFunction, LossFunction } from '../types';

export abstract class BaseFunction implements CriterionFunction {
    protected lossFunc: LossFunction;

    constructor(loss: LossFunction) {
        this.lossFunc = loss;
    }

    abstract impurity(yTrue: number[][]): number;

    /**
     * Computes the loss between true values and predicted values.
     *
     * @param yTrue - The true values (labels).
     * @param yPred - The predicted values.
     * @returns Scalar representing the computed loss.
     */
    loss(yTrue: Tensor2D, yPred: Tensor2D): Scalar {
        return this.lossFunc.compute(yTrue, yPred.toFloat());
    }

    /**
     * Disposes of any resources used by the loss function.
     */
    dispose(): void {
        this.lossFunc.dispose?.();
    }
}
