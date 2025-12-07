import type {
    ConfusionMatrixData,
    RegressionMetricsData,
    RocCurveData,
} from '@/app/shared/visualization';
import type { CriterionConfig } from '@/ml/factories';
import type { MatrixLike } from '@/ml/matrix';
import type {
    EnsembleTree,
    TreeCallbackParameters as TreeCallbackParametersType,
} from '@/ml/types';

export type TreeModelVariant = 'decision' | 'bagging' | 'forest' | 'extra';

export type TreeSettings = {
    type: 'tree';
    modelVariant: TreeModelVariant;
    criterion: CriterionConfig;
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    maxFeatures?: number;
    numRandomThresholds?: number;
    estimators?: number;
};

export type TreeRepresentation = {
    type: 'tree';
    representation: EnsembleTree;
};

export type TreeCallbackParameters = {
    type: 'tree';
    callbackParameters: TreeCallbackParametersType;
};

export type TreeClassificationTrainingReport = {
    type: 'tree';
    taskType: 'classification';
    iterations: number[];
    testAccuracy: number;
    trainAccuracy: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;

    trainConfusionMatrix: ConfusionMatrixData;
    testConfusionMatrix?: ConfusionMatrixData;

    trainRocCurve: RocCurveData;
    testRocCurve?: RocCurveData;
};

export type TreeRegressionTrainingReport = {
    type: 'tree';
    taskType: 'regression';
    iterations: number[];
    testLoss: number;
    trainPredictedLabels: MatrixLike;
    testPredictedLabels?: MatrixLike;
    predictionPredictedLabels?: MatrixLike;
    trainMetrics: RegressionMetricsData | null;
    testMetrics?: RegressionMetricsData | null;
    trainResiduals: MatrixLike;
    testResiduals?: MatrixLike;
};
