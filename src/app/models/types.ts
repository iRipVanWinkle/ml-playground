import type { Dataset, TaskType } from '@/app/shared/types';
import type { SystemSettings } from '@/app/features/configure-system';
import type { TransformationSettings } from '@/app/features/transform-data';
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
import type {
    NaiveBayesCallbackParameters,
    NaiveBayesRepresentation,
    NaiveBayesSettings,
    NaiveBayesTrainingReport,
} from './naive-bayes/types';

export type ModelSettings =
    | LinearSettings
    | LogisticSettings
    | NeuralSettings
    | TreeSettings
    | NaiveBayesSettings;

export type ModelType = ModelSettings['type'];

export type ModelRepresentation =
    | LinearRepresentation
    | LogisticRepresentation
    | NeuralRepresentation
    | TreeRepresentation
    | NaiveBayesRepresentation;

export type CallbackParameters =
    | LinearCallbackParameters
    | LogisticCallbackParameters
    | NeuralCallbackParameters
    | TreeCallbackParameters
    | NaiveBayesCallbackParameters;

export type TrainingReport =
    | LinearTrainingReport
    | LogisticTrainingReport
    | NeuralClassificationTrainingReport
    | NeuralRegressionTrainingReport
    | TreeClassificationTrainingReport
    | TreeRegressionTrainingReport
    | NaiveBayesTrainingReport;

export type TrainingSettings<TModelSettings extends ModelSettings = ModelSettings> = {
    taskType: TaskType;
    modelSettings: TModelSettings;
    systemSettings: SystemSettings;
    dataSettings: TransformationSettings;
    data: Dataset;
};
