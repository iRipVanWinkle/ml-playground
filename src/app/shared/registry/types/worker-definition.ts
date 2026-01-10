import type { ModelType, TrainingSettings } from '@/app/models/types';
import type { Model, TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '../../workers';
import type { TaskType } from '../../types';
import type { CallbackParametersOf, RepresentationOf, SettingsOf, TrainingReportOf } from './utils';

export interface WorkerDefinition<TKey extends ModelType = ModelType> {
    key: TKey;

    /*  Factory function to create model instance */
    modelFactory: (
        settings: TrainingSettings<SettingsOf<TKey>>,
        eventEmitter?: TrainingEventEmitter,
        trainingController?: TrainingControl,
    ) => Model<RepresentationOf<TKey>>;

    /*  Factory function to create live metrics instance */
    liveMetricsFactory: (
        model: Model<RepresentationOf<TKey>>,
        datasetManager: DatasetManager,
        taskType?: TaskType,
    ) => LiveMetrics<CallbackParametersOf<TKey>, TrainingReportOf<TKey>>;

    /* Extract the model parameters from the training report */
    extractParameters: (report: TrainingReportOf<TKey>) => RepresentationOf<TKey>;
}
