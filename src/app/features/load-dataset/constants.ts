import type { DataSectionState } from './types';

export const DEFAULT_STATE: DataSectionState = {
    file: null,
    shuffleData: true,
    trainTestSplit: 80,
};

export const PREPERED_REGRESSION_DATASETS = [
    {
        value: './data/world-happiness-report-2017 1(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita.)',
    },
    {
        value: './data/world-happiness-report-2017 2(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita., Freedom)',
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

export const PREPERED_CLASSIFICATION_DATASETS = [
    {
        value: './data/mnist-number-0-1.csv',
        label: 'MNIST numbers (0, 1)',
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

    {
        value: './data/iris-petal.csv',
        label: 'Iris Petal',
    },
    {
        value: './data/mnist-number.csv',
        label: 'MNIST numbers',
    },
    {
        value: './data/mnist-fashion.csv',
        label: 'MNIST fashion',
    },
];
