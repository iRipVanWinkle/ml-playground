import type { Tensor2D, Scalar } from '@tensorflow/tfjs';
import type {
    Model,
    NaiveBayesParams,
    PredictionMetadata,
    TrainingControl,
    TrainingEventEmitter,
} from '../../types';

export type BaseNaiveBayesOptions = {
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

export abstract class BaseNaiveBayes<T extends NaiveBayesParams> implements Model<T> {
    protected params?: T;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;

    constructor(options: BaseNaiveBayesOptions) {
        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;
    }

    abstract train(X: Tensor2D, y: Tensor2D): Promise<T>;
    abstract predict(X: Tensor2D, theta?: T | undefined): Tensor2D;
    abstract predictWithMetadata(X: Tensor2D, theta?: T): PredictionMetadata;
    abstract evaluate(
        X: Tensor2D,
        y: Tensor2D,
        theta?: T | undefined,
    ): [Tensor2D, Tensor2D, Scalar];

    dispose(): void {
        this.params = undefined;
    }

    /**
     * Returns the index of the maximum value in an array (argmax).
     * @param arr - Array of numbers (e.g., class probabilities or log-probabilities)
     * @returns Index of the maximum value
     */
    protected probabilityToClassIndex(arr: ArrayLike<number>): number {
        let maxIdx = 0;
        let maxVal = arr[0];
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
