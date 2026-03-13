import type { BaseTrainingReport } from '@/app/shared/types';
import type {
    GaussianDistributionCallbackParameters as GaussianDistributionCallbackParametersMl,
    GaussianDistributionParams,
} from '@/ml/types';
import type { MatrixLike } from '@/app/shared/helpers';

export type GaussianDistributionVariant = 'diagonal' | 'full';

export type GaussianDistributionSettings = {
    type: 'gaussian-distribution';
    variant: GaussianDistributionVariant;
    threshold: number;
    varianceSmoothing: number;
};

export type GaussianDistributionRepresentation = {
    type: 'gaussian-distribution';
    representation: GaussianDistributionParams;
};

export type GaussianDistributionCallbackParameters = {
    type: 'gaussian-distribution';
    callbackParameters: GaussianDistributionCallbackParametersMl;
};

export type GaussianDistributionTrainingReport = BaseTrainingReport & {
    type: 'gaussian-distribution';
    taskType: 'anomaly';
    trainAnomalyRate: number;
    testAnomalyRate?: number;
    trainPredictions: MatrixLike;
    testPredictions?: MatrixLike;
    params: GaussianDistributionParams;
};
