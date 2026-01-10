import type { ConfusionMatrixData, RocCurveData } from '@/app/shared/visualization';
import type { MatrixLike } from '@/app/shared/helpers';
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

export type NaiveBayesTrainingReport = {
    type: 'naive-bayes';
    taskType: 'classification';
    testAccuracy: number;
    trainAccuracy: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;
    iteration: number;
    params: NaiveBayesParams;

    trainConfusionMatrix: ConfusionMatrixData;
    testConfusionMatrix?: ConfusionMatrixData;

    trainRocCurve: RocCurveData;
    testRocCurve?: RocCurveData;
};
