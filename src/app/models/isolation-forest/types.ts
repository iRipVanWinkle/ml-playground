import type { BaseAnomalyReport } from '@/app/shared/types';
import type {
    IsolationForestCallbackParameters as IsolationForestCallbackParametersType,
    EnsembleTree,
    IsolationEnsembleTree,
} from '@/ml/types';

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

export type IsolationForestTrainingReport = BaseAnomalyReport & {
    type: 'isolation-forest';
    scoreThreshold: number;
    params: EnsembleTree;
};
