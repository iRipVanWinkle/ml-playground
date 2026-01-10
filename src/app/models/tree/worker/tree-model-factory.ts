import { criterionFactory } from '@/ml/factories';
import {
    BaggingClassifier,
    BaggingRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
} from '@/ml/models';
import type { TrainingSettings } from '../../types';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { TreeSettings } from '../types';

export function treeModelFactory(
    settings: TrainingSettings<TreeSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { modelSettings, taskType } = settings;
    const {
        modelVariant,
        criterion: criterionConfig,
        estimators,
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf,
        maxFeatures,
        numRandomThresholds,
    } = modelSettings;
    const isRegression = taskType === 'regression';

    const criterion = criterionFactory(criterionConfig);
    const commonParams = {
        criterion,
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf,
        eventEmitter,
        trainingController,
    };
    const ensembleParams = { ...commonParams, estimators };
    const forestParams = { ...ensembleParams, maxFeatures };

    let model;
    switch (modelVariant) {
        case 'bagging':
            model = new (isRegression ? BaggingRegressor : BaggingClassifier)(ensembleParams);
            break;
        case 'forest':
            model = new (isRegression ? RandomForestRegressor : RandomForestClassifier)(
                forestParams,
            );
            break;
        case 'extra':
            model = new (isRegression ? ExtraTreesRegressor : ExtraTreesClassifier)({
                ...forestParams,
                numRandomThresholds,
            });
            break;
        case 'decision':
        default:
            model = new (isRegression ? DecisionTreeRegressor : DecisionTreeClassifier)(
                commonParams,
            );
            break;
    }

    return model;
}
