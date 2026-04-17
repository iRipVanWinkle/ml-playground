import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { LinearRegressionPage } from './pages/LinearRegressionPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATASET_FILE = path.resolve(__dirname, './data/linear_regression_test_data.csv');

const EXPECTED_RESULTS = {
    default: {
        trainLoss: 'Train Loss: 3.9046',
        testLoss: 'Test Loss: 3.8844',
        metrics: {
            mse: '3.9046',
            rmse: '1.9760',
            mae: '1.6674',
            r2: '0.9329',
        },
        learnedParameters: {
            bias: '1.5924',
            weights: { x1: '+3.0013' },
        },
    },
    withNormalization: {
        trainLoss: 'Train Loss: 48.9290',
        testLoss: 'Test Loss: 27.9256',
        metrics: {
            mse: '48.9290',
            rmse: '6.9949',
            mae: '6.3933',
            r2: '0.1592',
        },
        learnedParameters: {
            bias: '11.0732',
            weights: { x1: '+4.8239' },
        },
    },
    withMAELoss: {
        trainLoss: 'Train Loss: 2.0015',
        testLoss: 'Test Loss: 2.2352',
        metrics: {
            mse: '5.8822',
            rmse: '2.4253',
            mae: '2.0015',
            r2: '0.8989',
        },
        learnedParameters: {
            bias: '0.7785',
            weights: { x1: '+3.0460' },
        },
    },
    withHuberLoss: {
        trainLoss: 'Train Loss: 1.8038',
        testLoss: 'Test Loss: 1.1603',
        metrics: {
            mse: '21.4229',
            rmse: '4.6285',
            mae: '3.8439',
            r2: '0.6319',
        },
        learnedParameters: {
            bias: '4.5221',
            weights: { x1: '+3.3555' },
        },
    },
    withSGD: {
        trainLoss: 'Train Loss: 3.8976',
        testLoss: 'Test Loss: 4.0784',
        metrics: {
            mse: '3.8976',
            rmse: '1.9742',
            mae: '1.6510',
            r2: '0.9330',
        },
        learnedParameters: {
            bias: '1.6113',
            weights: { x1: '+2.9641' },
        },
    },
    withMomentum: {
        trainLoss: 'Train Loss: 19.1801',
        testLoss: 'Test Loss: 10.4284',
        metrics: {
            mse: '19.1801',
            rmse: '4.3795',
            mae: '3.7459',
            r2: '0.6704',
        },
        learnedParameters: {
            bias: '5.2905',
            weights: { x1: '+3.2030' },
        },
    },
    withL2Regularization: {
        trainLoss: 'Train Loss: 3.4449',
        testLoss: 'Test Loss: 3.7214',
        metrics: {
            mse: '3.4449',
            rmse: '1.8560',
            mae: '1.5262',
            r2: '0.9408',
        },
        learnedParameters: {
            bias: '1.9445',
            weights: { x1: '+2.8672' },
        },
    },
} as const;

test.describe('Linear Regression Training', () => {
    let linearRegressionPage: LinearRegressionPage;

    test.beforeEach(async ({ page }) => {
        linearRegressionPage = new LinearRegressionPage(page, DATASET_FILE);
        await linearRegressionPage.navigateToPage();
        await linearRegressionPage.configureDataset();
        await linearRegressionPage.setBasicConfiguration();
    });

    test('should successfully train model by default', async () => {
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const defaultExpected = EXPECTED_RESULTS.default;
        await linearRegressionPage.verifyTrainingResults(
            defaultExpected.trainLoss,
            defaultExpected.testLoss,
        );
        await linearRegressionPage.verifyRegressionMetrics(defaultExpected.metrics);
        await linearRegressionPage.verifyLearnedParameters(defaultExpected.learnedParameters);
    });

    test('should successfully train model using Z-Score normalization', async () => {
        await linearRegressionPage.setNormalization('Z-Score');
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const normalizationExpected = EXPECTED_RESULTS.withNormalization;
        await linearRegressionPage.verifyTrainingResults(
            normalizationExpected.trainLoss,
            normalizationExpected.testLoss,
        );
        await linearRegressionPage.verifyRegressionMetrics(normalizationExpected.metrics);
        await linearRegressionPage.verifyLearnedParameters(normalizationExpected.learnedParameters);
    });

    test('should successfully train model using MAE loss function', async () => {
        await linearRegressionPage.setLossFunction('MAE (Mean Absolute Error)');
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const maeExpected = EXPECTED_RESULTS.withMAELoss;
        await linearRegressionPage.verifyTrainingResults(
            maeExpected.trainLoss,
            maeExpected.testLoss,
        );
        await linearRegressionPage.verifyRegressionMetrics(maeExpected.metrics);
        await linearRegressionPage.verifyLearnedParameters(maeExpected.learnedParameters);
    });

    test('should successfully train model using Huber loss function', async () => {
        await linearRegressionPage.setLossFunction('Huber', 0.5);
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const huberExpected = EXPECTED_RESULTS.withHuberLoss;
        await linearRegressionPage.verifyTrainingResults(
            huberExpected.trainLoss,
            huberExpected.testLoss,
        );
        await linearRegressionPage.verifyRegressionMetrics(huberExpected.metrics);
        await linearRegressionPage.verifyLearnedParameters(huberExpected.learnedParameters);
    });

    test('should successfully train model using Stochastic Gradient Descent', async () => {
        await linearRegressionPage.setOptimizer('Stochastic Gradient Descent', { batchSize: 8 });
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const sgdExpected = EXPECTED_RESULTS.withSGD;
        await linearRegressionPage.verifyTrainingResults(
            sgdExpected.trainLoss,
            sgdExpected.testLoss,
        );
        await linearRegressionPage.verifyRegressionMetrics(sgdExpected.metrics);
        await linearRegressionPage.verifyLearnedParameters(sgdExpected.learnedParameters);
    });

    test('should successfully train model using Momentum', async () => {
        await linearRegressionPage.setOptimizer('Momentum');
        await linearRegressionPage.setLearningRate(0.1);
        await linearRegressionPage.setMaxIterations(50);
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const momentumExpected = EXPECTED_RESULTS.withMomentum;
        await linearRegressionPage.verifyTrainingResults(
            momentumExpected.trainLoss,
            momentumExpected.testLoss,
        );
        await linearRegressionPage.verifyRegressionMetrics(momentumExpected.metrics);
        await linearRegressionPage.verifyLearnedParameters(momentumExpected.learnedParameters);
    });

    test('should successfully train model using L2 regularization', async () => {
        await linearRegressionPage.setRegularization('L2 (Ridge)');
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        const l2Expected = EXPECTED_RESULTS.withL2Regularization;
        await linearRegressionPage.verifyTrainingResults(l2Expected.trainLoss, l2Expected.testLoss);
        await linearRegressionPage.verifyRegressionMetrics(l2Expected.metrics);
        await linearRegressionPage.verifyLearnedParameters(l2Expected.learnedParameters);
    });
});
