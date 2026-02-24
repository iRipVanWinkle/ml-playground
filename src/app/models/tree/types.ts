import type { CriterionConfig } from '@/ml/factories';
import type { BaseClassificationReport, BaseRegressionReport } from '@/app/shared/types';
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

export type TreeClassificationTrainingReport = BaseClassificationReport & {
    type: 'tree';
    params: EnsembleTree;
};

export type TreeRegressionTrainingReport = BaseRegressionReport & {
    type: 'tree';
    params: EnsembleTree;
};
