import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { KNNPage } from './pages/KNNPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('KNN Training', () => {
    let knnPage: KNNPage;

    test.describe('Regression', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/linear_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainR2: 'Train R²: 0.9915',
                testR2: 'Test R²: 0.9579',
                metrics: {
                    mse: '0.4931',
                    rmse: '0.7022',
                    mae: '0.6227',
                    r2: '0.9915',
                },
            },
            withDistanceWeights: {
                trainR2: 'Train R²: 1.0000',
                testR2: 'Test R²: 0.9746',
                metrics: {
                    mse: '0.0000',
                    rmse: '0.0000',
                    mae: '0.0000',
                    r2: '1.0000',
                },
            },
            withManhattanK3: {
                trainR2: 'Train R²: 0.9951',
                testR2: 'Test R²: 0.9767',
                metrics: {
                    mse: '0.2851',
                    rmse: '0.5340',
                    mae: '0.4375',
                    r2: '0.9951',
                },
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            knnPage = new KNNPage(page, DATASET_FILE);
            await knnPage.navigateToRegressionPage();
            await knnPage.configureDataset();
            await knnPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await knnPage.verifyTrainingResults(defaultExpected.trainR2, defaultExpected.testR2);
            await knnPage.verifyRegressionMetrics(defaultExpected.metrics);
        });

        test('should successfully train model using distance weights', async () => {
            await knnPage.setWeights('Distance');

            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const distanceExpected = EXPECTED_RESULTS.withDistanceWeights;
            await knnPage.verifyTrainingResults(distanceExpected.trainR2, distanceExpected.testR2);
            await knnPage.verifyRegressionMetrics(distanceExpected.metrics);
        });

        test('should successfully train model using Manhattan distance with K=3', async () => {
            await knnPage.setK(3);
            await knnPage.setDistance('Manhattan');

            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const manhattanExpected = EXPECTED_RESULTS.withManhattanK3;
            await knnPage.verifyTrainingResults(
                manhattanExpected.trainR2,
                manhattanExpected.testR2,
            );
            await knnPage.verifyRegressionMetrics(manhattanExpected.metrics);
        });
    });

    test.describe('Binary Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
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
            withDistanceK3: {
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 80.00%',
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
            knnPage = new KNNPage(page, DATASET_FILE);
            await knnPage.navigateToClassificationPage();
            await knnPage.configureDataset();
            await knnPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await knnPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await knnPage.verifyConfusionMatrixMetrics(defaultExpected.confusionMatrixMetrics);
        });

        test('should successfully train model using distance weights with K=3', async () => {
            await knnPage.setK(3);
            await knnPage.setWeights('Distance');

            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const distanceExpected = EXPECTED_RESULTS.withDistanceK3;
            await knnPage.verifyTrainingResults(
                distanceExpected.trainAccuracy,
                distanceExpected.testAccuracy,
            );
            await knnPage.verifyConfusionMatrixMetrics(distanceExpected.confusionMatrixMetrics);
        });
    });

    test.describe('Multiclass Classification', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
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
            withDistanceManhattan: {
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
                multiclassROCAUC: { macroAuc: '1.000', weightedAuc: '1.000' },
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            knnPage = new KNNPage(page, DATASET_FILE);
            await knnPage.navigateToClassificationPage();
            await knnPage.configureDataset();
            await knnPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await knnPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await knnPage.verifyConfusionMatrixMetrics(defaultExpected.confusionMatrixMetrics);
            await knnPage.verifyMulticlassROCAUC(defaultExpected.multiclassROCAUC);
        });

        test('should successfully train model using distance weights with Manhattan distance', async () => {
            await knnPage.setWeights('Distance');
            await knnPage.setDistance('Manhattan');

            await knnPage.startTraining();
            await knnPage.waitForTrainingCompletion();

            const manhattanExpected = EXPECTED_RESULTS.withDistanceManhattan;
            await knnPage.verifyTrainingResults(
                manhattanExpected.trainAccuracy,
                manhattanExpected.testAccuracy,
            );
            await knnPage.verifyConfusionMatrixMetrics(manhattanExpected.confusionMatrixMetrics);
            await knnPage.verifyMulticlassROCAUC(manhattanExpected.multiclassROCAUC);
        });
    });
});
