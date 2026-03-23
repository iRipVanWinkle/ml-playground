import { fill, greater, tensor2d, tidy, where, type Tensor2D } from '@tensorflow/tfjs';
import { TrainingController } from '../../controllers/TrainingController';
import { EventEmitter } from '../../events/EventEmitter';
import type {
    AnomalyDetectionMetadata,
    IsolationEnsembleTree,
    Model,
    TrainingControl,
    TrainingEventEmitter,
    TreeNode,
} from '../../types';
import {
    IsolationSplitStrategy,
    bootstrapFeatures,
    subsampleFeatures,
    TreeBuilder,
    zeros,
} from '../../tree-builders';
import { assertModelTrained } from '../../utils';

export type IsolationForestOptions = {
    estimators?: number;
    maxSamples?: number;
    contamination?: number;
    bootstrap?: boolean;
    eventEmitter?: TrainingEventEmitter;
    trainingController?: TrainingControl;
};

/**
 * Isolation Forest — unsupervised anomaly detection algorithm.
 *
 * An ensemble of isolation trees is built by recursively partitioning
 * random subsamples of the training data with random splits.  Anomalies
 * are isolated near the root and therefore have shorter average path
 * lengths.  The anomaly score is derived from that average path length
 * normalised by the expected path length for a random BST.
 *
 * **Reference**: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).
 * Isolation Forest. *IEEE ICDM*.
 */
export class IsolationForest implements Model<IsolationEnsembleTree> {
    private estimators: number;
    private maxSamples: number;
    private contamination: number;
    private bootstrap: boolean;

    private eventEmitter?: TrainingEventEmitter;
    private trainingController?: TrainingControl;

    private treeBuilder: TreeBuilder;
    private trees: IsolationEnsembleTree = { trees: [], scoreThreshold: 0.5 };

    private actualMaxSamples = 0;

    constructor(options: IsolationForestOptions = {}) {
        this.estimators = options.estimators ?? 100;
        this.maxSamples = options.maxSamples ?? 256;
        this.contamination = options.contamination ?? 0.1;
        this.bootstrap = options.bootstrap ?? false;

        this.eventEmitter = options.eventEmitter;
        this.trainingController = options.trainingController;

        const dummyEmitter = new EventEmitter();
        const dummyTrainingController = new TrainingController(dummyEmitter);
        this.treeBuilder = new TreeBuilder(dummyEmitter, dummyTrainingController);
    }

    async train(X: Tensor2D): Promise<IsolationEnsembleTree> {
        const isSyncBackend = true;
        const numSamples = X.shape[0];
        this.actualMaxSamples = Math.min(this.maxSamples, numSamples);
        const maxDepth = Math.ceil(Math.log2(this.actualMaxSamples));
        const XArray = await X.array();

        const trees: TreeNode[] = [];

        // Step 1: Build an ensemble of isolation trees on random subsamples
        for (let i = 0; i < this.estimators; i++) {
            // Handle pause/step logic
            await this.trainingController?.handleControlFlow(isSyncBackend);
            if (this.trainingController?.isTrainingStopped) {
                break;
            }

            const [subsample, dummyTargets] = await this.prepareTrainingData(X, i);

            const splitStrategy = new IsolationSplitStrategy(i);

            const treeOptions = {
                splitStrategy,
                calculateNodeValueFn: (targets: number[][]) => ({ value: targets.length }),
                maxDepth,
                minSamplesSplit: 2,
                minSamplesLeaf: 1,
            };

            const tree = await this.treeBuilder.buildTree(subsample, dummyTargets, treeOptions, i);

            trees.push(tree);

            await this.emitCallback(i, trees, XArray);
        }

        // Step 2: Compute anomaly scores for all training samples
        // Higher scores are more anomalous; the (1 - contamination) quantile
        // ensures that approximately `contamination` fraction of training
        // samples exceed the threshold and are classified as anomalies.
        const trainingScores = this.computeRawScores(XArray, trees);

        // Step 3: Derive the decision threshold from the score distribution
        const scoreThreshold = quantile(trainingScores, 1 - this.contamination);

        this.trees = { trees, scoreThreshold };

        return this.trees;
    }

    predict(X: Tensor2D, ensemble?: IsolationEnsembleTree): Tensor2D {
        const ensembleTrees = ensemble ?? this.trees;
        assertModelTrained(ensembleTrees.trees);

        const threshold = ensembleTrees.scoreThreshold;

        return tidy(() => {
            const scores = this.scoreAnomaly(X, ensembleTrees.trees);
            return this.scoresToPredictions(scores, threshold);
        });
    }

    predictWithMetadata(X: Tensor2D, ensemble?: IsolationEnsembleTree): AnomalyDetectionMetadata {
        const ensembleTrees = ensemble ?? this.trees;
        assertModelTrained(ensembleTrees.trees);

        const threshold = ensembleTrees.scoreThreshold;

        const scores = this.scoreAnomaly(X, ensembleTrees.trees);
        const predictions = this.scoresToPredictions(scores, threshold);

        return {
            type: 'anomaly-detection',
            predictions,
            probabilities: scores,
            dispose() {
                predictions.dispose();
                scores.dispose();
            },
        };
    }

    dispose(): void {
        this.trees = { trees: [], scoreThreshold: 0.5 };
    }

    private scoreAnomaly(X: Tensor2D, trees: ReadonlyArray<TreeNode>): Tensor2D {
        const XArray = X.arraySync();
        const rawScores = this.computeRawScores(XArray, trees);

        return tensor2d(new Float32Array(rawScores), [rawScores.length, 1]);
    }

    private computeRawScores(XArray: number[][], trees: ReadonlyArray<TreeNode>): number[] {
        const normFactor = expectedPathLength(this.actualMaxSamples);

        return XArray.map((sample) => {
            const meanPathLen =
                trees.reduce((sum, root) => sum + pathLength(sample, root), 0) / trees.length;

            return normFactor > 0 ? Math.pow(2, -meanPathLen / normFactor) : 0.5;
        });
    }

    private scoresToPredictions(scores: Tensor2D, threshold: number): Tensor2D {
        return tidy(() => {
            const shape = scores.shape;

            const negOnes = fill(shape, -1);
            const posOnes = fill(shape, 1);
            const thresholds = fill(shape, threshold);
            const isAnomaly = greater(scores, thresholds);

            const predictions = where(isAnomaly, negOnes, posOnes);

            return predictions as Tensor2D;
        });
    }

    private async prepareTrainingData(X: Tensor2D, index = 0): Promise<[number[][], number[][]]> {
        const subsample = this.bootstrap
            ? bootstrapFeatures(X, this.actualMaxSamples, index)
            : subsampleFeatures(X, this.actualMaxSamples, index);

        const subsampleArray = await subsample.array();
        const dummyTargetsArray = zeros([subsample.shape[0], 1]);

        subsample.dispose();

        return [subsampleArray, dummyTargetsArray];
    }

    private async emitCallback(step: number, trees: TreeNode[], XArray: number[][]): Promise<void> {
        if (this.eventEmitter) {
            // This calculation only exist for the live metrics and is not needed for the actual training.
            const trainingScores = this.computeRawScores(XArray, trees);
            const scoreThreshold = quantile(trainingScores, 1 - this.contamination);

            await this.eventEmitter.emit('callback', {
                threadId: step,
                iteration: step + 1,
                ensemble: { trees, scoreThreshold },
            });
        }
    }
}

/** Euler–Mascheroni constant used in the average path-length correction. */
export const EULER_CONSTANT = 0.5772156649;

/**
 * Expected average path length of an unsuccessful search in a Binary Search
 * Tree with `n` elements — used to normalise anomaly scores.
 */
export function expectedPathLength(n: number): number {
    if (n <= 1) return 0;
    if (n === 2) return 1;
    return 2 * (Math.log(n - 1) + EULER_CONSTANT) - (2 * (n - 1)) / n;
}

/**
 * Traverse `node` for `sample` and return the path length, including the
 * correction term for the leaf node size.
 */
export function pathLength(sample: number[], node: TreeNode): number {
    let depth = 0;
    let current: TreeNode = node;

    while (current.threshold !== null) {
        const goLeft = sample[current.featureIndex!] < current.threshold;
        const next = goLeft ? current.leftChild : current.rightChild;

        if (!next) break;

        current = next;
        depth++;
    }

    // Add the correction factor for the remaining samples in the leaf.
    return depth + expectedPathLength(current.value);
}

/**
 * Compute the `q`-th quantile (0 ≤ q ≤ 1) of a numeric array using linear
 * interpolation between adjacent sorted values.
 */
export function quantile(values: number[], q: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const idx = q * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
}
