import { type Tensor2D } from '@tensorflow/tfjs';
import type { EnsembleAggregatorFn, EnsembleTree } from '../../types';
import { BaseDecisionTree, type DecisionTreeOptions } from './BaseDecisionTree';
import { bootstrapSample, type TreeBuilderOptions } from '../../tree-builders';

export type BaseEnsembleOptions = DecisionTreeOptions & {
    bootstrap?: boolean;
    estimators?: number;
    aggregator?: EnsembleAggregatorFn;
};

export abstract class BaseEnsembleTree extends BaseDecisionTree {
    protected bootstrap: boolean;
    protected estimators: number;
    protected aggregator: EnsembleAggregatorFn;

    constructor(options: BaseEnsembleOptions) {
        super(options);

        this.bootstrap = options.bootstrap ?? true;
        this.estimators = options.estimators ?? 10;
        this.aggregator = options.aggregator ?? ((v) => v as Tensor2D);
    }

    protected async trainTreeEnsemble(
        X: Tensor2D,
        y: Tensor2D,
        options: TreeBuilderOptions,
    ): Promise<EnsembleTree> {
        const treePromises = [];

        for (let index = 0; index < this.estimators; index++) {
            // Bootstrap sampling
            const [XArray, yArray] = await this.prepareTrainingData(X, y, index);

            // Build the tree for this estimator
            const treePromise = this.treeBuilder.buildTree(XArray, yArray, options, index);

            treePromises.push(treePromise);
        }

        this.trees = await Promise.all(treePromises);

        return this.trees;
    }

    protected async prepareTrainingData(
        X: Tensor2D,
        y: Tensor2D,
        index: number = 0,
    ): Promise<[number[][], number[][]]> {
        const [XFeatures, yTargets] = this.bootstrap
            ? bootstrapSample(X, y, index)
            : [X.clone(), y.clone()];

        const [XArray, yArray] = await super.prepareTrainingData(XFeatures, yTargets);

        XFeatures.dispose();
        yTargets.dispose();

        return [XArray, yArray];
    }
}
