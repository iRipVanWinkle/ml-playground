import { useAppStore } from './store';

// Task
export const useTaskType = () => useAppStore((s) => s.taskType);

// Dataset
export const useDataset = () => useAppStore((s) => s.dataset);
export const useNumTrainInputFeatures = () =>
    useAppStore((s) => s.dataset.trainInputFeatures[0]?.length ?? 0);
export const useNumCategories = () => useAppStore((s) => s.dataset.categories?.length ?? 0);
export const useHasData = () => useAppStore((s) => s.dataset.trainInputFeatures.length > 0);

// Model Settings
export const useModelSettings = () => useAppStore((s) => s.modelSettings);
export const useModelType = () => useAppStore((s) => s.modelSettings.type);

// Transformations
export const useTransformations = () => useAppStore((s) => s.transformations.transformations);
export const useNormalization = () => useAppStore((s) => s.transformations.normalization);

// System
export const useBackend = () => useAppStore((s) => s.system.backend);
export const useRandomSeed = () => useAppStore((s) => s.system.randomSeed);

// Training Control
export const useTrainingState = () => useAppStore((s) => s.training.state);
export const useIsTraining = () =>
    useAppStore((s) => s.training.state === 'training' || s.training.state === 'paused');

// Training Report
export const useTrainingReport = () => useAppStore((s) => s.trainingReport);
