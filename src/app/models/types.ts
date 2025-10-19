import type { TaskType } from '@/app/shared/types';
import type { SystemSettings } from '@/app/features/system-settings';
import type { TransformationSettings } from '@/app/features/transform-data';
import type { DataState } from '@/app/features/load-dataset';
import type {
    LinearCallbackParameters,
    LinearSettings,
    LinearRepresentation,
    LinearTrainingReport,
} from './linear/types';
import type {
    LogisticCallbackParameters,
    LogisticRepresentation,
    LogisticSettings,
    LogisticTrainingReport,
} from './logistic/types';
import type {
    NeuralCallbackParameters,
    NeuralClassificationTrainingReport,
    NeuralRegressionTrainingReport,
    NeuralRepresentation,
    NeuralSettings,
} from './neural/types';
import type {
    TreeCallbackParameters,
    TreeClassificationTrainingReport,
    TreeRegressionTrainingReport,
    TreeRepresentation,
    TreeSettings,
} from './tree/types';

export type ModelSettings = LinearSettings | LogisticSettings | NeuralSettings | TreeSettings;

export type ModelType = ModelSettings['type'];

export type ModelRepresentation =
    | LinearRepresentation
    | LogisticRepresentation
    | NeuralRepresentation
    | TreeRepresentation;

export type CallbackParameters =
    | LinearCallbackParameters
    | LogisticCallbackParameters
    | NeuralCallbackParameters
    | TreeCallbackParameters;

export type TrainingReport =
    | LinearTrainingReport
    | LogisticTrainingReport
    | NeuralClassificationTrainingReport
    | NeuralRegressionTrainingReport
    | TreeClassificationTrainingReport
    | TreeRegressionTrainingReport;

export type TrainingSettings<TModelSettings extends ModelSettings = ModelSettings> = {
    taskType: TaskType;
    modelSettings: TModelSettings;
    systemSettings: SystemSettings;
    dataSettings: TransformationSettings;
    data: DataState;
};
