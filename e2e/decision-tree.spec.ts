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
                confusionMatrixMetrics: {
                    Accuracy: '100.0%',
                    MCC: '1.000',
                    "Cohen's Kappa": '1.000',
                    Precision: '100.0%',
                    Recall: '100.0%',
                    F1: '100.0%',
                },
            },
            withRandomForest: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 90.00%',
                confusionMatrixMetrics: {
                    Accuracy: '100.0%',
                    MCC: '1.000',
                    "Cohen's Kappa": '1.000',
                    Precision: '100.0%',
                    Recall: '100.0%',
                    F1: '100.0%',
                },
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

            const defaultExpected = EXPECTED_RESULTS.default;
            await decisionTreePage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await decisionTreePage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
        });

        test('should successfully train model with Random Forest variant', async () => {
            await decisionTreePage.setModelVariant('Random Forest');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            const rfExpected = EXPECTED_RESULTS.withRandomForest;
            await decisionTreePage.verifyTrainingResults(
                rfExpected.trainAccuracy,
                rfExpected.testAccuracy,
            );
            await decisionTreePage.verifyConfusionMatrixMetrics(rfExpected.confusionMatrixMetrics);
        });
    });

    test.describe('Regression', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/linear_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainR2: 'Train R²: 0.9990',
                testR2: 'Test R²: 0.9744',
                metrics: {
                    mse: '0.0609',
                    rmse: '0.2467',
                    mae: '0.1415',
                    r2: '0.9990',
                },
            },
            withMAE: {
                trainR2: 'Train R²: 0.9987',
                testR2: 'Test R²: 0.9717',
                metrics: {
                    mse: '0.0738',
                    rmse: '0.2716',
                    mae: '0.1441',
                    r2: '0.9987',
                },
            },
            withHuber: {
                trainR2: 'Train R²: 0.9988',
                testR2: 'Test R²: 0.9735',
                metrics: {
                    mse: '0.0687',
                    rmse: '0.2621',
                    mae: '0.1320',
                    r2: '0.9988',
                },
            },
            withRandomForest: {
                trainR2: 'Train R²: 0.9963',
                testR2: 'Test R²: 0.9665',
                metrics: {
                    mse: '0.2173',
                    rmse: '0.4661',
                    mae: '0.3129',
                    r2: '0.9963',
                },
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

            const defaultExpected = EXPECTED_RESULTS.default;
            await decisionTreePage.verifyTrainingResults(
                defaultExpected.trainR2,
                defaultExpected.testR2,
            );
            await decisionTreePage.verifyRegressionMetrics(defaultExpected.metrics);
        });

        test('should successfully train model with MAE criterion', async () => {
            await decisionTreePage.setCriterion('MAE (Mean Absolute Error)');
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            const maeExpected = EXPECTED_RESULTS.withMAE;
            await decisionTreePage.verifyTrainingResults(maeExpected.trainR2, maeExpected.testR2);
            await decisionTreePage.verifyRegressionMetrics(maeExpected.metrics);
        });

        test('should successfully train model with Huber criterion', async () => {
            await decisionTreePage.setCriterion('Huber', 0.5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            const huberExpected = EXPECTED_RESULTS.withHuber;
            await decisionTreePage.verifyTrainingResults(
                huberExpected.trainR2,
                huberExpected.testR2,
            );
            await decisionTreePage.verifyRegressionMetrics(huberExpected.metrics);
        });

        test('should successfully train model with Random Forest variant', async () => {
            await decisionTreePage.setModelVariant('Random Forest');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            const rfExpected = EXPECTED_RESULTS.withRandomForest;
            await decisionTreePage.verifyTrainingResults(rfExpected.trainR2, rfExpected.testR2);
            await decisionTreePage.verifyRegressionMetrics(rfExpected.metrics);
        });
    });

    test.describe('Multiclass Classification', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 91.67%',
                confusionMatrixMetrics: {
                    Accuracy: '100.0%',
                    MCC: '1.000',
                    "Cohen's Kappa": '1.000',
                    'Macro Precision': '100.0%',
                    'Macro Recall': '100.0%',
                    'Macro F1': '100.0%',
                    'Weighted Precision': '100.0%',
                    'Weighted Recall': '100.0%',
                    'Weighted F1': '100.0%',
                },
            },
            withBagging: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 100.00%',
                confusionMatrixMetrics: {
                    Accuracy: '100.0%',
                    MCC: '1.000',
                    "Cohen's Kappa": '1.000',
                    'Macro Precision': '100.0%',
                    'Macro Recall': '100.0%',
                    'Macro F1': '100.0%',
                    'Weighted Precision': '100.0%',
                    'Weighted Recall': '100.0%',
                    'Weighted F1': '100.0%',
                },
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

            const defaultExpected = EXPECTED_RESULTS.default;
            await decisionTreePage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await decisionTreePage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
        });

        test('should successfully train model with Bagging variant', async () => {
            await decisionTreePage.setModelVariant('Bagging');
            await decisionTreePage.setEstimators(5);
            await decisionTreePage.startTraining();
            await decisionTreePage.waitForTrainingCompletion();

            const baggingExpected = EXPECTED_RESULTS.withBagging;
            await decisionTreePage.verifyTrainingResults(
                baggingExpected.trainAccuracy,
                baggingExpected.testAccuracy,
            );
            await decisionTreePage.verifyConfusionMatrixMetrics(
                baggingExpected.confusionMatrixMetrics,
            );
        });
    });
});
