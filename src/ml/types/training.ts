import type { Tensor2D } from '@tensorflow/tfjs';
import type { EventEmitter } from '../events/EventEmitter';
import type {
    TreeNode,
    IsolationEnsembleTree,
    NaiveBayesParams,
    KNNParams,
    GaussianDistributionParams,
    DBSCANParams,
    HierarchicalClusteringParams,
} from './data';

export type TreeCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    tree: TreeNode;
    threadName?: string;
}>;

export type IsolationForestCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    ensemble: IsolationEnsembleTree;
    threadName?: string;
}>;

export type OptimizerCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    alfa: number;
    loss: number;
    theta: Tensor2D;
    threadName?: string;
}>;

export type NaiveBayesCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    params: NaiveBayesParams;
}>;

export type KMeansCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    centroids: Tensor2D;
    assignments: Tensor2D;
    inertia: number;
}>;

export type HierarchicalClusteringCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    assignments: Int32Array;
    numClusters: number;
    params?: HierarchicalClusteringParams;
}>;

export type KNNCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    params: KNNParams;
}>;

export type GaussianDistributionCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    params: GaussianDistributionParams;
}>;

export type DBSCANCallbackParameters = Readonly<{
    threadId: number;
    iteration: number;
    threadName?: string;
    assignments: Int32Array;
    numClusters: number;
    activePointIndex?: number;
    epsilon: number;
    params?: DBSCANParams;
}>;

export type CallbackParameters =
    | OptimizerCallbackParameters
    | TreeCallbackParameters
    | IsolationForestCallbackParameters
    | NaiveBayesCallbackParameters
    | KMeansCallbackParameters
    | KNNCallbackParameters
    | GaussianDistributionCallbackParameters
    | DBSCANCallbackParameters
    | HierarchicalClusteringCallbackParameters;

export type TrainingState = 'transforming' | 'training' | 'paused' | 'stopped' | 'stepped-forward';

/**
 * Emits events during the training lifecycle. Models emit 'state' when
 * training state changes, 'callback' with iteration data for progress
 * tracking, and 'error'/'info' for diagnostics.
 */
export interface TrainingEventEmitter extends EventEmitter {
    on(event: 'state', listener: (state: TrainingState) => void): void;
    on(event: 'callback', listener: (params: CallbackParameters) => void): void;
    on(event: 'error', listener: (message: string) => void): void;
    on(event: 'info', listener: (message: string) => void): void;

    emit(event: 'state', state: TrainingState): Promise<void>;
    emit(event: 'callback', params: CallbackParameters): Promise<void>;
    emit(event: 'error', message: string): Promise<void>;
    emit(event: 'info', message: string): Promise<void>;
}

/**
 * Controls the training process from outside the training loop.
 * Used by the UI to pause, resume, step through, or stop training.
 * The handleControlFlow() method is called inside the training loop
 * to check for pending control signals.
 */
export interface TrainingControl {
    stop(): void;
    pause(): void;
    resume(): void;
    step(): void;
    get isTrainingStopped(): boolean;
    handleControlFlow(isSyncBackend?: boolean): Promise<void>;
}
