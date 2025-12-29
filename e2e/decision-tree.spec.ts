import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { DecisionTreePage } from './pages/DecisionTreePage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Decision Tree Training', () => {
    let decisionTreePage: DecisionTreePage;

    test.describe('Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 60.00%',
            },
            withEntropy: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 60.00%',
            },
            withRandomForest: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 90.00%',
            },
            withBagging: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 70.00%',
            },
            withExtraTrees: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 80.00%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            decisionTreePage = new DecisionTreePage(page, DATASET_FILE);
            await decisionTreePage.navigateToClassificationPage();
            await decisionTreePage.configureDataset();
            await decisionTreePage.setBasicConfiguration();
        });

        test('should successfully train model with default Gini criterion', async () => {
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
        });

        test('should successfully train model with Entropy criterion', async () => {
            await decisionTreePage.setCriterion('Entropy');
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withEntropy.trainAccuracy,
                EXPECTED_RESULTS.withEntropy.testAccuracy,
            );
        });

        test('should successfully train model with Random Forest variant', async () => {
            await decisionTreePage.setModelVariant('Random Forest');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withRandomForest.trainAccuracy,
                EXPECTED_RESULTS.withRandomForest.testAccuracy,
            );
        });

        test('should successfully train model with Bagging variant', async () => {
            await decisionTreePage.setModelVariant('Bagging');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withBagging.trainAccuracy,
                EXPECTED_RESULTS.withBagging.testAccuracy,
            );
        });

        test('should successfully train model with Extra Trees variant', async () => {
            await decisionTreePage.setModelVariant('Extra Trees');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withExtraTrees.trainAccuracy,
                EXPECTED_RESULTS.withExtraTrees.testAccuracy,
            );
        });
    });

    test.describe('Regression', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/linear_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainLoss: 'Train Loss: --',
                testLoss: 'Test Loss: 0.8199',
            },
            withMAE: {
                trainLoss: 'Train Loss: --',
                testLoss: 'Test Loss: 0.7978',
            },
            withHuber: {
                trainLoss: 'Train Loss: --',
                testLoss: 'Test Loss: 0.2758',
            },
            withRandomForest: {
                trainLoss: 'Train Loss: --',
                testLoss: 'Test Loss: 1.0752',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            decisionTreePage = new DecisionTreePage(page, DATASET_FILE);
            await decisionTreePage.navigateToRegressionPage();
            await decisionTreePage.configureDataset();
            await decisionTreePage.setBasicConfiguration();
        });

        test('should successfully train model with default MSE criterion', async () => {
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainLoss,
                EXPECTED_RESULTS.default.testLoss,
            );
        });

        test('should successfully train model with MAE criterion', async () => {
            await decisionTreePage.setCriterion('MAE (Mean Absolute Error)');
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withMAE.trainLoss,
                EXPECTED_RESULTS.withMAE.testLoss,
            );
        });

        test('should successfully train model with Huber criterion', async () => {
            await decisionTreePage.setCriterion('Huber', 0.5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withHuber.trainLoss,
                EXPECTED_RESULTS.withHuber.testLoss,
            );
        });

        test('should successfully train model with Random Forest variant', async () => {
            await decisionTreePage.setModelVariant('Random Forest');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withRandomForest.trainLoss,
                EXPECTED_RESULTS.withRandomForest.testLoss,
            );
        });
    });

    test.describe('Multiclass Classification', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 97.92%',
                testAccuracy: 'Test Accuracy: 91.67%',
            },
            withBagging: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 100.00%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            decisionTreePage = new DecisionTreePage(page, DATASET_FILE);
            await decisionTreePage.navigateToClassificationPage();
            await decisionTreePage.configureDataset();
            await decisionTreePage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
        });

        test('should successfully train model with Bagging variant', async () => {
            await decisionTreePage.setModelVariant('Bagging');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            await decisionTreePage.verifyTrainingResults(
                EXPECTED_RESULTS.withBagging.trainAccuracy,
                EXPECTED_RESULTS.withBagging.testAccuracy,
            );
        });
    });
});
