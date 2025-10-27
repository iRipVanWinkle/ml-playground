import type { Dataset, TaskType } from '@/app/shared/types';

export type DataSectionState = {
    file: File | null;
    datasetPath?: string;
    shuffleData: boolean;
    trainTestSplit: number;
    isImageDataset?: boolean;
};

export type ExtractFeaturesOptions = {
    file: File;
    shuffleData?: boolean;
    trainTestSplit?: number;
    taskType?: TaskType;
    seed?: number;
};

export type DataState = {
    dataset: Dataset;
};
