import type { ModelType, TrainingSettings } from '@/app/models/types';
import type { Model, TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { PipelineModel } from '@/ml/models';
import type { DatasetManager, LiveMetrics } from '../../workers';
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
        model: PipelineModel<RepresentationOf<TKey>>,
        datasetManager: DatasetManager,
        settings: TrainingSettings<SettingsOf<TKey>>,
    ) => LiveMetrics<CallbackParametersOf<TKey>, TrainingReportOf<TKey>>;

    /* Extract the model parameters from the training report */
    extractParameters: (report: TrainingReportOf<TKey>) => RepresentationOf<TKey> | null;
}
