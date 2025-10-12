import type { TaskType } from '@/app/store';

export type DataSectionState = {
    file: File | null;
    datasetPath?: string;
    shuffleData: boolean;
    trainTestSplit: number;
};

export type DataSectionProps = {
    disabled?: boolean;
};

export type ExtractFeaturesOptions = {
    file: File;
    shuffleData?: boolean;
    trainTestSplit?: number;
    taskType?: TaskType;
};

export type DataState = {
    trainInputFeatures: number[][];
    trainTargetLabels: number[][];
    testInputFeatures: number[][];
    testTargetLabels: number[][];
    predictionInputFeatures?: number[][];
    xMin: number[];
    xMax: number[];
    headers: string[];
    categories?: string[];
};
