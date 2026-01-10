import type { ModelType, TrainingSettings } from '@/app/models/types';
import type { Model, TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '../../workers';
import type { TaskType } from '../../types';
import type { CallbackParametersOf, RepresentationOf, SettingsOf, TrainingReportOf } from './utils';

export interface WorkerDefinition<TKey extends ModelType = ModelType> {
    key: TKey;

    modelFactory: (
        settings: TrainingSettings<SettingsOf<TKey>>,
        eventEmitter?: TrainingEventEmitter,
        trainingController?: TrainingControl,
    ) => Model<RepresentationOf<TKey>>;

    liveMetricsFactory: (
        model: Model<RepresentationOf<TKey>>,
        datasetManager: DatasetManager,
        taskType?: TaskType,
    ) => LiveMetrics<CallbackParametersOf<TKey>, TrainingReportOf<TKey>>;
}
