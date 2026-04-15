import { useAppStore } from './store';

export const {
    switchTask,

    setDataset,

    setNormalization,
    setTransformations,

    setModelType,
    updateModelSettings,

    setBackend,
    setRandomSeed,

    setTrainingState,
    setTrainingReport,

    snapshotTrainingSettings,
} = useAppStore.getState();
