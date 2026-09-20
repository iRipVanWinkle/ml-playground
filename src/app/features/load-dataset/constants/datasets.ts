import type { TaskType } from '@/app/shared/types';
import type { DataSectionState } from '../types';

export type DatasetOption = {
    value: string;
    label: string;
    hint?: string;
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
        hint: '1 feature',
    },
    {
        value: './data/world-happiness-report-2017 2(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita., Freedom)',
        hint: '2 features',
    },
    {
        value: './data/bodyfat.csv',
        label: 'Body Fat Prediction',
        hint: 'tabular',
    },
    {
        value: './data/california-housing.csv',
        label: 'California Housing Prices',
        hint: 'tabular',
    },
    {
        value: './data/non-linear-regression.csv',
        label: 'Non linear regression',
        hint: 'synthetic',
    },
    {
        value: './data/linear-relationship.csv',
        label: 'Linear relationship',
        hint: 'synthetic',
    },
    {
        value: './data/quadratic-relationship.csv',
        label: 'Quadratic relationship',
        hint: 'synthetic',
    },
    {
        value: './data/wave-pattern-regression.csv',
        label: 'Wave pattern regression',
        hint: 'synthetic',
    },
];

export const PREPARED_CLASSIFICATION_DATASETS: DatasetOption[] = [
    {
        value: './data/microchips-tests.csv',
        label: 'Microchips Tests (non linear)',
        hint: 'synthetic',
    },
    {
        value: './data/circle-classification.csv',
        label: 'Circle classification',
        hint: 'synthetic',
    },
    {
        value: './data/cluster-2d.csv',
        label: 'Cluster 2D',
        hint: 'synthetic',
    },
    {
        value: './data/spiral.csv',
        label: 'Spiral',
        hint: 'synthetic',
    },
    {
        value: './data/XOR.csv',
        label: 'XOR',
        hint: 'synthetic',
    },
    {
        value: './data/mnist-number-0-1.csv',
        label: 'MNIST numbers (0, 1)',
        isImage: true,
        hint: 'image',
    },
    {
        value: './data/breast_cancer_wisconsin.csv',
        label: 'Breast cancer Wisconsin (Diagnostic)',
        hint: 'tabular',
    },
    {
        value: './data/iris-petal.csv',
        label: 'Iris (Petals)',
        hint: 'tabular',
    },
    {
        value: './data/iris.csv',
        label: 'Iris',
        hint: 'tabular',
    },
    {
        value: './data/winequality-red.csv',
        label: 'Wine quality (Red)',
        hint: 'tabular',
    },
    {
        value: './data/winequality-white.csv',
        label: 'Wine quality (White)',
        hint: 'tabular',
    },
    {
        value: './data/mnist-number.csv',
        label: 'MNIST numbers',
        isImage: true,
        hint: 'image',
    },
    {
        value: './data/mnist-fashion.csv',
        label: 'MNIST fashion',
        isImage: true,
        hint: 'image',
    },
    {
        value: './data/emotions.csv',
        label: 'Emotions (48x48)',
        isImage: true,
        hint: 'image',
    },
];

export const PREPARED_ANOMALY_DATASETS: DatasetOption[] = [
    {
        value: './data/server-operational-params.csv',
        label: 'Server Operational Parameters',
        hint: 'tabular',
    },
    {
        value: './data/server-operational-params-big.csv',
        label: 'Server Operational Parameters Big (10k rows)',
        hint: 'tabular',
    },
];

export const PREPARED_CLUSTERING_DATASETS: DatasetOption[] = [
    {
        value: './data/iris-petal.csv',
        label: 'Iris (Petals)',
        hint: 'tabular',
    },
    {
        value: './data/iris.csv',
        label: 'Iris',
        hint: 'tabular',
    },
    {
        value: './data/microchips-tests.csv',
        label: 'Microchips Tests (non linear)',
        hint: 'synthetic',
    },
    {
        value: './data/circle-classification.csv',
        label: 'Circle classification',
        hint: 'synthetic',
    },
    {
        value: './data/cluster-2d.csv',
        label: 'Cluster 2D',
        hint: 'synthetic',
    },
    {
        value: './data/spiral.csv',
        label: 'Spiral',
        hint: 'synthetic',
    },
    {
        value: './data/XOR.csv',
        label: 'XOR',
        hint: 'synthetic',
    },
];

export function getDatasetsForTask(taskType: TaskType): DatasetOption[] {
    switch (taskType) {
        case 'regression':
            return PREPARED_REGRESSION_DATASETS;
        case 'classification':
            return PREPARED_CLASSIFICATION_DATASETS;
        case 'clustering':
            return PREPARED_CLUSTERING_DATASETS;
        case 'anomaly':
            return PREPARED_ANOMALY_DATASETS;
        default:
            return [];
    }
}
