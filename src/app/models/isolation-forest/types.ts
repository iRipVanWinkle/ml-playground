import type { BaseTrainingReport } from '@/app/shared/types';
import type {
    IsolationForestCallbackParameters as IsolationForestCallbackParametersType,
    EnsembleTree,
    IsolationEnsembleTree,
} from '@/ml/types';
import type { MatrixLike } from '@/app/shared/helpers';

export type IsolationForestSettings = {
    type: 'isolation-forest';
    estimators: number;
    maxSamples: number;
    contamination: number;
    bootstrap: boolean;
};

export type IsolationForestRepresentation = {
    type: 'isolation-forest';
    representation: IsolationEnsembleTree;
};

export type IsolationForestCallbackParameters = {
    type: 'isolation-forest';
    callbackParameters: IsolationForestCallbackParametersType;
};

export type IsolationForestTrainingReport = BaseTrainingReport & {
    type: 'isolation-forest';
    taskType: 'anomaly';
    trainAnomalyRate: number;
    testAnomalyRate?: number;
    scoreThreshold: number;
    trainPredictions: MatrixLike;
    testPredictions?: MatrixLike;
    params: EnsembleTree;
};
