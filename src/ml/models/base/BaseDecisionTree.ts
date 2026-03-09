import { type Tensor2D } from '@tensorflow/tfjs';
import type {
    CriterionFunction,
    EnsembleTree,
    Model,
    PredictionMetadata,
    TrainingControl,
    TrainingEventEmitter,
} from '../../types';
import { TreeBuilder } from '../../tree-builders';

export type DecisionTreeOptions = {
    criterion: CriterionFunction;
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

export abstract class BaseDecisionTree implements Model<EnsembleTree> {
    protected criterion: CriterionFunction;
    protected maxDepth?: number;
    protected minSamplesSplit: number;
    protected minSamplesLeaf: number;
    protected eventEmitter?: TrainingEventEmitter;
    protected trainingController?: TrainingControl;

    protected treeBuilder: TreeBuilder;
    protected trees: EnsembleTree = [];

    constructor(options: DecisionTreeOptions) {
        this.criterion = options.criterion;
        this.maxDepth = options.maxDepth;
        this.minSamplesSplit = options.minSamplesSplit ?? 2;
        this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;

        this.treeBuilder = new TreeBuilder(options.eventEmitter, options.trainingController);
    }

    abstract train(X: Tensor2D, y: Tensor2D): Promise<EnsembleTree>;

    abstract predict(X: Tensor2D, trees?: EnsembleTree): Tensor2D;

    abstract predictWithMetadata(X: Tensor2D, trees?: EnsembleTree): PredictionMetadata;

    dispose(): void {
        this.trees = [];
    }

    protected async prepareTrainingData(
        X: Tensor2D,
        y: Tensor2D,
    ): Promise<[number[][], number[][]]> {
        return Promise.all([X.array(), y.array()]);
    }
}
