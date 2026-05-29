import type { TrainingSettings, TrainingReport } from '@/app/models/types';
import type { TypedArray } from '@/app/shared/helpers';
import type { TaskType } from '@/app/shared/types';
import type { WorkerManager } from '@/app/shared/workers/manager';

export type PredictionResult = {
    type: TaskType;
    prediction: number;
    probabilities?: TypedArray;
};

export interface PredictionWorkerManager extends WorkerManager<UIToWorkerMessage, TrainingReport> {
    on(type: 'predictions', handler: (predictions: PredictionResult) => void): () => void;
    on(type: 'error', handler: (error: Error) => void): () => void;
    on(type: 'info', handler: (info: string) => void): () => void;
    on(type: 'finished', handler: () => void): () => void;
}

export type UIToWorkerMessage = PredictMessage & { sentAt?: number };

export type PredictMessage = {
    type: 'predict';
    payload: TrainingSettings & { example: number[]; report: TrainingReport };
    requestId?: string;
};
