import type {
    AnyTransformation,
    Dataset,
    NormalizationMethod,
    SystemSettings,
    TaskType,
    TrainingState,
    TransformationSettings,
    UserExample,
} from '@/app/shared/types';
import type {
    ModelSettings,
    ModelType,
    TrainingReport,
    TrainingSettings,
} from '@/app/models/types';

type UserResult = {
    prediction: number;
    probabilities?: number[];
};

export type AppState = {
    taskType: TaskType;
    dataset: Dataset;
    modelSettings: ModelSettings;
    transformations: TransformationSettings;
    system: SystemSettings;
    training: {
        state: TrainingState;
    };
    trainingReport: TrainingReport;
    userExample: UserExample;
};

export type AppActions = {
    updateModelSettings: (patch: Partial<Omit<ModelSettings, 'type'>>) => void;
    setModelType: (modelType: ModelType) => void;
    resetModelSettings: (
        modelType: ModelType,
        taskType: TaskType,
    ) => Pick<AppState, 'modelSettings'>;
    setTransformations: (transformations: AnyTransformation[]) => void;
    setNormalization: (normalization: NormalizationMethod) => void;
    resetTransformations: () => Pick<AppState, 'transformations'>;
    setBackend: (backend: SystemSettings['backend']) => void;
    setRandomSeed: (randomSeed: SystemSettings['randomSeed']) => void;
    setTrainingState: (state: TrainingState) => void;
    setTrainingReport: (report: TrainingReport) => void;
    resetTrainingReport: (
        modelType?: ModelType,
        taskType?: TaskType,
    ) => Pick<AppState, 'trainingReport'>;
    resetTrainingControls: () => Pick<AppState, 'training'>;

    switchTask: (taskType: TaskType) => void;
    setDataset: (dataset: Dataset) => void;

    snapshotTrainingSettings: () => TrainingSettings;

    setUserExampleInputs: (inputs: number[]) => void;
    setUserExamplePrediction: (result: UserResult) => void;
    resetUserExample: () => void;
};

export type AppStore = AppState & AppActions;
