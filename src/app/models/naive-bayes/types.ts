import type { BaseClassificationReport } from '@/app/shared/types';
import type {
    NaiveBayesCallbackParameters as NaiveBayesCallbackParametersType,
    NaiveBayesParams,
} from '@/ml/types';

export type NaiveBayesVariant = 'gaussian' | 'quadratic';

export type NaiveBayesSettings = {
    type: 'naive-bayes';
    variant: NaiveBayesVariant;
};

export type NaiveBayesRepresentation = {
    type: 'naive-bayes';
    representation: NaiveBayesParams;
};

export type NaiveBayesCallbackParameters = {
    type: 'naive-bayes';
    callbackParameters: NaiveBayesCallbackParametersType;
};

export type NaiveBayesTrainingReport = BaseClassificationReport & {
    type: 'naive-bayes';
    iteration: number;
    params: NaiveBayesParams;
};
