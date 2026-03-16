import type { TrainingControl, TrainingEventEmitter, TreeNode } from '../types';
import type { SplitStrategy, TreeBuilderOptions } from './types';
import { gather, zeros } from '../utils';

type RootNodeRef = {
    current: TreeNode | null;
};

interface NodeData {
    readonly features: number[][];
    readonly targets: number[][];
    readonly indices: number[];
}

interface PendingNode {
    readonly nodeData: NodeData;
    readonly depth: number;
    readonly parent: TreeNode | null;
    readonly isLeftChild: boolean;
}

type TreeBuilderContext = TreeBuilderOptions & {
    rootRef: RootNodeRef;
    splitStrategy: SplitStrategy;
    minSamplesSplit: number;
    minSamplesLeaf: number;
};

const DEFAULT_CONTEXT = {
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    threadId: 0,
};

export class TreeBuilder {
    private context: TreeBuilderContext | null = null;
    private eventEmitter?: TrainingEventEmitter;
    private trainingController?: TrainingControl;

    constructor(eventEmitter?: TrainingEventEmitter, trainingController?: TrainingControl) {
        this.eventEmitter = eventEmitter;
        this.trainingController = trainingController;
    }

    async buildTree(
        X: number[][],
        y: number[][],
        options: TreeBuilderOptions,
        threadId = 0,
    ): Promise<TreeNode> {
        const rootRef: RootNodeRef = { current: null };

        const context = {
            ...DEFAULT_CONTEXT,
            ...options,
            threadId,
            rootRef,
        };

        // Initialize the tree building process
        const treeBuildingIterator = this.createTreeIterator(X, y, context);

        for await (const iteration of treeBuildingIterator) {
            await this.eventEmitter?.emit('callback', {
                threadId,
                iteration,
                tree: rootRef.current!,
            });
        }

        return rootRef.current!;
    }

    /**
     * Create an async iterator for the tree building process
     */
    private async *createTreeIterator(
        X: number[][],
        y: number[][],
        context: TreeBuilderContext,
    ): AsyncGenerator<number, void, unknown> {
        const isSyncBackend = true;
        // Create the root node data
        const indices = Array.from({ length: X.length }, (_, i) => i);
        const rootNodeData: NodeData = { features: X, targets: y, indices };

        // Add the root node to the pending queue
        const pendingNodes = [
            {
                nodeData: rootNodeData,
                depth: 0,
                parent: null,
                isLeftChild: false,
            },
        ];

        let stepCount = 0;

        while (pendingNodes.length !== 0) {
            // Handle pause/step logic similar to BaseOptimizer
            await this.trainingController?.handleControlFlow(isSyncBackend);

            if (this.trainingController?.isTrainingStopped) {
                break;
            }

            this.context = context;

            this.processNextStep(pendingNodes);

            this.context = null;

            yield stepCount++; // Yield the current step count
        }
    }

    private processNextStep(pendingNodes: PendingNode[]): { depth: number } | void {
        if (pendingNodes.length === 0) {
            return;
        }

        const currentNodeInfo = pendingNodes.shift()!;
        const { nodeData, depth, parent, isLeftChild } = currentNodeInfo;
        const { features, targets, indices } = nodeData;
        const numSamples = indices.length;

        // Check stopping criteria
        if (this.shouldStopSplitting(numSamples, depth)) {
            const leafNode = this.createLeafNode(targets);

            this.updateParentReference(parent, leafNode, isLeftChild);
            return;
        }

        // Find the best split
        console.time(`Find best split: depth=${depth}, numSamples=${numSamples}`);
        const bestSplit = this.context!.splitStrategy.findBestSplit(indices, features, targets);
        console.timeEnd(`Find best split: depth=${depth}, numSamples=${numSamples}`);

        if (!bestSplit) {
            const leafNode = this.createLeafNode(targets);

            this.updateParentReference(parent, leafNode, isLeftChild);

            return;
        }

        const { calculateNodeValueFn } = this.context!;

        // Create internal node
        const internalNode: TreeNode = {
            featureIndex: bestSplit.featureIndex,
            threshold: bestSplit.threshold,
            leftChild: null,
            rightChild: null,
            ...calculateNodeValueFn(targets),
            // samples: numSamples,
            // impurity: (bestSplit.leftImpurity + bestSplit.rightImpurity) / 2
        };

        // Update parent references
        this.updateParentReference(parent, internalNode, isLeftChild);

        // Create child node data
        const leftNodeData = this.createChildNodeData(features, targets, bestSplit.leftIndices);
        const rightNodeData = this.createChildNodeData(features, targets, bestSplit.rightIndices);

        // Add child nodes to the pending queue (process left child first)
        pendingNodes.push({
            nodeData: leftNodeData,
            depth: depth + 1,
            parent: internalNode,
            isLeftChild: true,
        });

        pendingNodes.push({
            nodeData: rightNodeData,
            depth: depth + 1,
            parent: internalNode,
            isLeftChild: false,
        });

        return {
            depth,
        };
    }

    private updateParentReference(
        parent: TreeNode | null,
        child: TreeNode,
        isLeftChild: boolean,
    ): void {
        const { rootRef } = this.context!;

        // Update root if this was the root node
        if (parent === null) {
            rootRef.current = child;

            return;
        }

        if (isLeftChild) {
            parent.leftChild = child;
        } else {
            parent.rightChild = child;
        }
    }

    private shouldStopSplitting(numSamples: number, depth: number): boolean {
        const { maxDepth, minSamplesSplit, minSamplesLeaf } = this.context!;

        return (
            (maxDepth !== undefined && depth >= maxDepth) ||
            numSamples < 2 * minSamplesLeaf ||
            numSamples < minSamplesSplit ||
            !!this.trainingController?.isTrainingStopped
        );
    }

    private createLeafNode(targets: number[][]): TreeNode {
        const { calculateNodeValueFn } = this.context!;

        return {
            featureIndex: null,
            threshold: null,
            leftChild: null,
            rightChild: null,
            ...calculateNodeValueFn(targets),
        };
    }

    private createChildNodeData(
        features: number[][],
        targets: number[][],
        indices: number[],
    ): NodeData {
        if (indices.length === 0) {
            return {
                features: zeros([0, features[0].length]),
                targets: zeros([0, targets[0].length]),
                indices: [],
            };
        }

        const childFeatures = gather(features, indices);
        const childTargets = gather(targets, indices);

        return {
            features: childFeatures,
            targets: childTargets,
            indices: Array.from({ length: indices.length }, (_, i) => i),
        };
    }
}
