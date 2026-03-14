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
import type {
    KMeansCallbackParameters,
    KMeansRepresentation,
    KMeansSettings,
    KMeansTrainingReport,
} from './k-means/types';
import type {
    KNNCallbackParameters,
    KNNClassificationTrainingReport,
    KNNRegressionTrainingReport,
    KNNRepresentation,
    KNNSettings,
} from './knn/types';
import type {
    GaussianDistributionCallbackParameters,
    GaussianDistributionRepresentation,
    GaussianDistributionSettings,
    GaussianDistributionTrainingReport,
} from './gaussian-distribution/types';
import type {
    DBSCANCallbackParameters,
    DBSCANRepresentation,
    DBSCANSettings,
    DBSCANTrainingReport,
} from './dbscan/types';
import type {
    IsolationForestCallbackParameters,
    IsolationForestRepresentation,
    IsolationForestSettings,
    IsolationForestTrainingReport,
} from './isolation-forest/types';

export type ModelSettings =
    | LinearSettings
    | LogisticSettings
    | NeuralSettings
    | TreeSettings
    | NaiveBayesSettings
    | KMeansSettings
    | KNNSettings
    | GaussianDistributionSettings
    | DBSCANSettings
    | IsolationForestSettings;

export type ModelType = ModelSettings['type'];

export type ModelRepresentation =
    | LinearRepresentation
    | LogisticRepresentation
    | NeuralRepresentation
    | TreeRepresentation
    | NaiveBayesRepresentation
    | KMeansRepresentation
    | KNNRepresentation
    | GaussianDistributionRepresentation
    | DBSCANRepresentation
    | IsolationForestRepresentation;

export type CallbackParameters =
    | LinearCallbackParameters
    | LogisticCallbackParameters
    | NeuralCallbackParameters
    | TreeCallbackParameters
    | NaiveBayesCallbackParameters
    | KMeansCallbackParameters
    | KNNCallbackParameters
    | GaussianDistributionCallbackParameters
    | DBSCANCallbackParameters
    | IsolationForestCallbackParameters;

export type TrainingReport =
    | LinearTrainingReport
    | LogisticTrainingReport
    | NeuralClassificationTrainingReport
    | NeuralRegressionTrainingReport
    | TreeClassificationTrainingReport
    | TreeRegressionTrainingReport
    | NaiveBayesTrainingReport
    | KMeansTrainingReport
    | KNNClassificationTrainingReport
    | KNNRegressionTrainingReport
    | GaussianDistributionTrainingReport
    | DBSCANTrainingReport
    | IsolationForestTrainingReport;

export type TrainingSettings<TModelSettings extends ModelSettings = ModelSettings> = {
    taskType: TaskType;
    modelSettings: TModelSettings;
    systemSettings: SystemSettings;
    dataSettings: TransformationSettings;
    dataset: Dataset;
};
