import type { TrainingSettings } from '@/app/models/types';
import type { WorkerManager } from '@/app/shared/workers';
import type { TrainingReport } from '../store';

export interface TrainingWorkerManager extends WorkerManager<UIToWorkerMessage, TrainingReport> {
    on(type: 'report', handler: (report: ArrayBufferLike) => void): () => void;
    on(type: 'state', handler: (state: string) => void): () => void;
    on(type: 'error', handler: (error: Error) => void): () => void;
    on(type: 'info', handler: (info: string) => void): () => void;
    on(type: 'finished', handler: () => void): void;
}

export type UIToWorkerMessage =
    | TrainMessage
    | TrainByStepMessage
    | StopMessage
    | PauseMessage
    | ResumeMessage
    | StepForwardMessage;

export type TrainMessage = {
    type: 'train';
    payload: TrainingSettings;
    requestId?: string;
};

export type TrainByStepMessage = {
    type: 'train-by-step';
    payload: TrainingSettings;
    requestId?: string;
};

export type StopMessage = {
    type: 'stop';
    payload?: never;
    requestId?: string;
};

export type PauseMessage = {
    type: 'pause';
    payload?: never;
    requestId?: string;
};

export type ResumeMessage = {
    type: 'resume';
    payload?: never;
    requestId?: string;
};

export type StepForwardMessage = {
    type: 'step-forward';
    payload?: never;
    requestId?: string;
};
