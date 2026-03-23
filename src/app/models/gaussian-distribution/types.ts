import type { BaseAnomalyReport } from '@/app/shared/types';
import type {
    GaussianDistributionCallbackParameters as GaussianDistributionCallbackParametersMl,
    GaussianDistributionParams,
} from '@/ml/types';

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

export type GaussianDistributionTrainingReport = BaseAnomalyReport & {
    type: 'gaussian-distribution';
    params: GaussianDistributionParams;
};
