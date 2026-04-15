import type { DataSectionState } from '../types';

type DatasetOption = {
    value: string;
    label: string;
    isImage?: boolean;
};

export const DEFAULT_STATE: DataSectionState = {
    file: null,
    shuffleData: true,
    trainTestSplit: 80,
};

export const PREPARED_REGRESSION_DATASETS: DatasetOption[] = [
    {
        value: './data/world-happiness-report-2017 1(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita.)',
    },
    {
        value: './data/world-happiness-report-2017 2(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita., Freedom)',
    },
    {
        value: './data/bodyfat.csv',
        label: 'Body Fat Prediction',
    },
    {
        value: './data/california-housing.csv',
        label: 'California Housing Prices',
    },
    {
        value: './data/non-linear-regression.csv',
        label: 'Non linear regression',
    },
    {
        value: './data/linear-relationship.csv',
        label: 'Linear relationship',
    },
    {
        value: './data/quadratic-relationship.csv',
        label: 'Quadratic relationship',
    },
    {
        value: './data/wave-pattern-regression.csv',
        label: 'Wave pattern regression',
    },
];

export const PREPARED_CLASSIFICATION_DATASETS: DatasetOption[] = [
    {
        value: './data/microchips-tests.csv',
        label: 'Microchips Tests (non linear)',
    },
    {
        value: './data/circle-classification.csv',
        label: 'Circle classification',
    },
    {
        value: './data/cluster-2d.csv',
        label: 'Cluster 2D',
    },
    {
        value: './data/spiral.csv',
        label: 'Spiral',
    },
    {
        value: './data/XOR.csv',
        label: 'XOR',
    },
    {
        value: './data/mnist-number-0-1.csv',
        label: 'MNIST numbers (0, 1)',
        isImage: true,
    },
    {
        value: './data/breast_cancer_wisconsin.csv',
        label: 'Breast cancer Wisconsin (Diagnostic)',
    },
    {
        value: './data/iris-petal.csv',
        label: 'Iris (Petals)',
    },
    {
        value: './data/iris.csv',
        label: 'Iris',
    },
    {
        value: './data/winequality-red.csv',
        label: 'Wine quality (Red)',
    },
    {
        value: './data/winequality-white.csv',
        label: 'Wine quality (White)',
    },
    {
        value: './data/mnist-number.csv',
        label: 'MNIST numbers',
        isImage: true,
    },
    {
        value: './data/mnist-fashion.csv',
        label: 'MNIST fashion',
        isImage: true,
    },
    {
        value: './data/emotions.csv',
        label: 'Emotions (48x48)',
        isImage: true,
    },
];

export const PREPARED_ANOMALY_DATASETS: DatasetOption[] = [
    {
        value: './data/server-operational-params.csv',
        label: 'Server Operational Parameters',
    },
    {
        value: './data/server-operational-params-big.csv',
        label: 'Server Operational Parameters Big (10k rows)',
    },
];

export const PREPARED_CLUSTERING_DATASETS: DatasetOption[] = [
    {
        value: './data/iris-petal.csv',
        label: 'Iris (Petals)',
    },
    {
        value: './data/iris.csv',
        label: 'Iris',
    },
    {
        value: './data/microchips-tests.csv',
        label: 'Microchips Tests (non linear)',
    },
    {
        value: './data/circle-classification.csv',
        label: 'Circle classification',
    },
    {
        value: './data/cluster-2d.csv',
        label: 'Cluster 2D',
    },
    {
        value: './data/spiral.csv',
        label: 'Spiral',
    },
    {
        value: './data/XOR.csv',
        label: 'XOR',
    },
];
