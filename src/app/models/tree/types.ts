import type { CriterionConfig } from '@/ml/factories';
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
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    predictionPredictedLabels?: number[][];
};

export type TreeRegressionTrainingReport = {
    type: 'tree';
    taskType: 'regression';
    iterations: number[];
    testLoss: number;
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    predictionPredictedLabels?: number[][];
};
