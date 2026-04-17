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
                confusionMatrixMetrics: {
                    Accuracy: '90.0%',
                    MCC: '0.800',
                    "Cohen's Kappa": '0.796',
                    Precision: '83.3%',
                    Recall: '93.8%',
                    F1: '88.2%',
                },
                multiclassROCAUC: { macroAuc: '0.919', weightedAuc: '0.926' },
            },
            quadratic: {
                trainAccuracy: 'Train Accuracy: 97.50%',
                testAccuracy: 'Test Accuracy: 80.00%',
                confusionMatrixMetrics: {
                    Accuracy: '97.5%',
                    MCC: '0.949',
                    "Cohen's Kappa": '0.947',
                    Precision: '100.0%',
                    Recall: '93.8%',
                    F1: '96.8%',
                },
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

            const gaussianExpected = EXPECTED_RESULTS.gaussian;
            await naiveBayesPage.verifyTrainingResults(
                gaussianExpected.trainAccuracy,
                gaussianExpected.testAccuracy,
            );
            await naiveBayesPage.verifyConfusionMatrixMetrics(
                gaussianExpected.confusionMatrixMetrics,
            );
            await naiveBayesPage.verifyMulticlassROCAUC(gaussianExpected.multiclassROCAUC);
        });

        test('should successfully train model with Quadratic variant', async () => {
            await naiveBayesPage.setVariant('Quadratic');
            await naiveBayesPage.startTraining();
            await naiveBayesPage.waitForTrainingCompletion();

            const quadraticExpected = EXPECTED_RESULTS.quadratic;
            await naiveBayesPage.verifyTrainingResults(
                quadraticExpected.trainAccuracy,
                quadraticExpected.testAccuracy,
            );
            await naiveBayesPage.verifyConfusionMatrixMetrics(
                quadraticExpected.confusionMatrixMetrics,
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
                confusionMatrixMetrics: {
                    Accuracy: '95.8%',
                    MCC: '0.937',
                    "Cohen's Kappa": '0.937',
                    'Macro Precision': '96.2%',
                    'Macro Recall': '96.2%',
                    'Macro F1': '96.2%',
                    'Weighted Precision': '95.8%',
                    'Weighted Recall': '95.8%',
                    'Weighted F1': '95.8%',
                },
                multiclassROCAUC: { macroAuc: '0.992', weightedAuc: '0.991' },
            },
            quadratic: {
                trainAccuracy: 'Train Accuracy: 93.75%',
                testAccuracy: 'Test Accuracy: 91.67%',
                confusionMatrixMetrics: {
                    Accuracy: '93.8%',
                    MCC: '0.906',
                    "Cohen's Kappa": '0.905',
                    'Macro Precision': '94.4%',
                    'Macro Recall': '94.2%',
                    'Macro F1': '94.3%',
                    'Weighted Precision': '93.8%',
                    'Weighted Recall': '93.8%',
                    'Weighted F1': '93.7%',
                },
                multiclassROCAUC: { macroAuc: '0.994', weightedAuc: '0.994' },
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

            const gaussianExpected = EXPECTED_RESULTS.gaussian;
            await naiveBayesPage.verifyTrainingResults(
                gaussianExpected.trainAccuracy,
                gaussianExpected.testAccuracy,
            );
            await naiveBayesPage.verifyConfusionMatrixMetrics(
                gaussianExpected.confusionMatrixMetrics,
            );
            await naiveBayesPage.verifyMulticlassROCAUC(gaussianExpected.multiclassROCAUC);
        });

        test('should successfully train model with Quadratic variant', async () => {
            await naiveBayesPage.setVariant('Quadratic');
            await naiveBayesPage.startTraining();
            await naiveBayesPage.waitForTrainingCompletion();

            const quadraticExpected = EXPECTED_RESULTS.quadratic;
            await naiveBayesPage.verifyTrainingResults(
                quadraticExpected.trainAccuracy,
                quadraticExpected.testAccuracy,
            );
            await naiveBayesPage.verifyConfusionMatrixMetrics(
                quadraticExpected.confusionMatrixMetrics,
            );
            await naiveBayesPage.verifyMulticlassROCAUC(quadraticExpected.multiclassROCAUC);
        });
    });
});
