import { expect, test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { LogisticRegressionPage } from './pages/LogisticRegressionPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Logistic Regression Training', () => {
    let logisticRegressionPage: LogisticRegressionPage;

    test.describe('Binary Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 90.00%',
                testAccuracy: 'Test Accuracy: 90.00%',
            },
            withTransformation: {
                trainAccuracy: '87.50%',
                testAccuracy: '100.00%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            logisticRegressionPage = new LogisticRegressionPage(page, DATASET_FILE);
            await logisticRegressionPage.navigateToPage();
            await logisticRegressionPage.configureDataset();
            await logisticRegressionPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
        });

        test('should successfully train model using transformation', async ({ page }) => {
            await logisticRegressionPage.setNormalization('Z-Score');
            await logisticRegressionPage.addTransformation('Polynomial', { degree: 3 });

            await expect(page.getByText('Output features: 7')).toBeVisible();

            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.withTransformation.trainAccuracy,
                EXPECTED_RESULTS.withTransformation.testAccuracy,
            );
        });
    });

    test.describe('Multiclass Classification (Softmax)', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: '68.75%',
                testAccuracy: '58.33%',
            },
            withTransformation: {
                trainAccuracy: '95.83%',
                testAccuracy: '100.00%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            logisticRegressionPage = new LogisticRegressionPage(page, DATASET_FILE);
            await logisticRegressionPage.navigateToPage();
            await logisticRegressionPage.configureDataset();
            await logisticRegressionPage.setBasicConfiguration();
        });

        test('should be disabled binary classification and enabled softmax', async ({ page }) => {
            await expect(page.getByTestId('classification-type-binary')).toBeDisabled();
            await expect(page.getByTestId('classification-type-softmax')).toBeChecked();
        });

        test('should successfully train model by default', async () => {
            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
        });

        test('should successfully train model using Momentum', async () => {
            await logisticRegressionPage.setOptimizer('Momentum');
            await logisticRegressionPage.setLearningRate(0.1);
            await logisticRegressionPage.setMaxIterations(50);
            await logisticRegressionPage.setRegularization('L2 (Ridge)', { lambda: 0.1 });

            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.withTransformation.trainAccuracy,
                EXPECTED_RESULTS.withTransformation.testAccuracy,
            );
        });
    });

    test.describe('Multiclass Classification (One-vs-Rest)', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: '68.75%',
                testAccuracy: '58.33%',
            },
            withTransformation: {
                trainAccuracy: '100.00%',
                testAccuracy: '91.67%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            logisticRegressionPage = new LogisticRegressionPage(page, DATASET_FILE);
            await logisticRegressionPage.navigateToPage();
            await logisticRegressionPage.configureDataset();
            await logisticRegressionPage.setBasicConfiguration();

            await page.getByTestId('classification-type-ovr').click();
        });

        test('should be disabled binary classification', async ({ page }) => {
            await expect(page.getByTestId('classification-type-binary')).toBeDisabled();
        });

        test('should successfully train model by default', async () => {
            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
        });

        test('should successfully train model using Momentum', async () => {
            await logisticRegressionPage.setOptimizer('Momentum');
            await logisticRegressionPage.setLearningRate(0.1);
            await logisticRegressionPage.setMaxIterations(50);
            await logisticRegressionPage.setRegularization('L2 (Ridge)', { lambda: 0.1 });

            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.withTransformation.trainAccuracy,
                EXPECTED_RESULTS.withTransformation.testAccuracy,
            );
        });
    });
});
