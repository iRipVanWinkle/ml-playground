import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { LinearRegressionPage } from './pages/LinearRegressionPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATASET_FILE = path.resolve(__dirname, './data/linear_regression_test_data.csv');

const EXPECTED_RESULTS = {
    default: {
        trainLoss: '3.7827',
        testLoss: '3.4807',
    },
    withNormalization: {
        trainLoss: '43.0870',
        testLoss: '71.3567',
    },
    withMAELoss: {
        trainLoss: '2.0948',
        testLoss: '1.6584',
    },
    withHuberLoss: {
        trainLoss: '0.7058',
        testLoss: '0.8800',
    },
    withSGD: {
        trainLoss: '2.5073',
        testLoss: '3.4927',
    },
    withScheduler: {
        trainLoss: '2.3750',
        testLoss: '2.3139',
    },
    withMomentum: {
        trainLoss: '7.0554',
        testLoss: '8.0749',
    },
    withL2Regularization: {
        trainLoss: '7.4047',
        testLoss: '3.1478',
    },
    withCustomWeightInitialization: {
        trainLoss: '19.4746',
        testLoss: '12.8995',
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

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.default.trainLoss,
            EXPECTED_RESULTS.default.testLoss,
        );
    });

    test('should successfully train model using Z-Score normalization', async () => {
        await linearRegressionPage.setNormalization('Z-Score');
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withNormalization.trainLoss,
            EXPECTED_RESULTS.withNormalization.testLoss,
        );
    });

    test('should successfully train model using MAE loss function', async () => {
        await linearRegressionPage.setLossFunction('MAE (Mean Absolute Error)');
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withMAELoss.trainLoss,
            EXPECTED_RESULTS.withMAELoss.testLoss,
        );
    });

    test('should successfully train model using Huber loss function', async () => {
        await linearRegressionPage.setLossFunction('Huber', 0.5);
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withHuberLoss.trainLoss,
            EXPECTED_RESULTS.withHuberLoss.testLoss,
        );
    });

    test('should successfully train model using Stochastic Gradient Descent', async () => {
        await linearRegressionPage.setOptimizer('Stochastic Gradient Descent', { batchSize: 8 });
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withSGD.trainLoss,
            EXPECTED_RESULTS.withSGD.testLoss,
        );
    });

    test('should successfully train model using Learning Rate Scheduler', async () => {
        await linearRegressionPage.setLearningRate(0.1, { s: 1, p: 0.5 });
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withScheduler.trainLoss,
            EXPECTED_RESULTS.withScheduler.testLoss,
        );
    });

    test('should successfully train model using Momentum', async () => {
        await linearRegressionPage.setOptimizer('Momentum');
        await linearRegressionPage.setLearningRate(0.1);
        await linearRegressionPage.setMaxIterations(50);
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withMomentum.trainLoss,
            EXPECTED_RESULTS.withMomentum.testLoss,
        );
    });

    test('should successfully train model using L2 regularization', async () => {
        await linearRegressionPage.setRegularization('L2 (Ridge)');
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withL2Regularization.trainLoss,
            EXPECTED_RESULTS.withL2Regularization.testLoss,
        );
    });

    test('should successfully train model using custom weight initialization', async () => {
        await linearRegressionPage.setWeightInitialization('Constant', { constant: '-100' });
        await linearRegressionPage.startTraining();
        await linearRegressionPage.waitForTrainingCompletion();

        await linearRegressionPage.verifyTrainingResults(
            EXPECTED_RESULTS.withCustomWeightInitialization.trainLoss,
            EXPECTED_RESULTS.withCustomWeightInitialization.testLoss,
        );
    });
});
