import type { ModelType, TrainingSettings } from '@/app/models/types';
import type { Model, TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { RepresentationOf, SettingsOf } from './utils';

export interface WorkerDefinition<TKey extends ModelType = ModelType> {
    key: TKey;

    modelFactory: (
        settings: TrainingSettings<SettingsOf<TKey>>,
        eventEmitter: TrainingEventEmitter,
        trainingController: TrainingControl,
    ) => Model<RepresentationOf<TKey>>;
}
