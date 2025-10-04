export type DataSectionState = {
    file: File | null;
    datasetPath?: string;
    shuffleData: boolean;
    trainTestSplit: number;
};

export type DataSectionProps = {
    disabled?: boolean;
};
