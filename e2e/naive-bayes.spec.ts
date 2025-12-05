import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { NaiveBayesPage } from './pages/NaiveBayesPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Naive Bayes Training', () => {
    let naiveBayesPage: NaiveBayesPage;

    test.describe('Binary Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            gaussian: {
                trainAccuracy: 'Train Accuracy: 90.00%',
                testAccuracy: 'Test Accuracy: 60.00%',
            },
            quadratic: {
                trainAccuracy: 'Train Accuracy: 92.50%',
                testAccuracy: 'Test Accuracy: 70.00%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            naiveBayesPage = new NaiveBayesPage(page, DATASET_FILE);
            await naiveBayesPage.navigateToPage();
            await naiveBayesPage.configureDataset();
            await naiveBayesPage.setBasicConfiguration();
        });

        test('should successfully train model with Gaussian variant', async () => {
            await naiveBayesPage.setVariant('Gaussian');
            await naiveBayesPage.startTraining();
            await naiveBayesPage.waitForTrainingCompletion();

            await naiveBayesPage.verifyTrainingResults(
                EXPECTED_RESULTS.gaussian.trainAccuracy,
                EXPECTED_RESULTS.gaussian.testAccuracy,
            );
        });

        test('should successfully train model with Quadratic variant', async () => {
            await naiveBayesPage.setVariant('Quadratic');
            await naiveBayesPage.startTraining();
            await naiveBayesPage.waitForTrainingCompletion();

            await naiveBayesPage.verifyTrainingResults(
                EXPECTED_RESULTS.quadratic.trainAccuracy,
                EXPECTED_RESULTS.quadratic.testAccuracy,
            );
        });
    });

    test.describe('Multiclass Classification', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            gaussian: {
                trainAccuracy: 'Train Accuracy: 95.83%',
                testAccuracy: 'Test Accuracy: 91.67%',
            },
            quadratic: {
                trainAccuracy: 'Train Accuracy: 95.83%',
                testAccuracy: 'Test Accuracy: 91.67%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            naiveBayesPage = new NaiveBayesPage(page, DATASET_FILE);
            await naiveBayesPage.navigateToPage();
            await naiveBayesPage.configureDataset();
            await naiveBayesPage.setBasicConfiguration();
        });

        test('should successfully train model with Gaussian variant', async () => {
            await naiveBayesPage.setVariant('Gaussian');
            await naiveBayesPage.startTraining();
            await naiveBayesPage.waitForTrainingCompletion();

            await naiveBayesPage.verifyTrainingResults(
                EXPECTED_RESULTS.gaussian.trainAccuracy,
                EXPECTED_RESULTS.gaussian.testAccuracy,
            );
        });

        test('should successfully train model with Quadratic variant', async () => {
            await naiveBayesPage.setVariant('Quadratic');
            await naiveBayesPage.startTraining();
            await naiveBayesPage.waitForTrainingCompletion();

            await naiveBayesPage.verifyTrainingResults(
                EXPECTED_RESULTS.quadratic.trainAccuracy,
                EXPECTED_RESULTS.quadratic.testAccuracy,
            );
        });
    });
});
