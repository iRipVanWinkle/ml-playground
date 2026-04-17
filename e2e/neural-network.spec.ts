import { expect, test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { NeuralNetworkPage } from './pages/NeuralNetworkPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Neural Network Training', () => {
    let neuralNetworkPage: NeuralNetworkPage;

    test.describe('Regression', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/linear_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainLoss: 'Train Loss: 5.5953',
                testLoss: 'Test Loss: 5.7880',
                metrics: {
                    mse: '5.5953',
                    rmse: '2.3654',
                    mae: '1.9956',
                    r2: '0.9039',
                },
            },
            withTransformation: {
                trainLoss: 'Train Loss: 1.1848',
                testLoss: 'Test Loss: 1.0362',
                metrics: {
                    mse: '1.1848',
                    rmse: '1.0885',
                    mae: '0.9129',
                    r2: '0.9796',
                },
            },
            withLayers: {
                trainLoss: 'Train Loss: 18.8793',
                testLoss: 'Test Loss: 9.9741',
                metrics: {
                    mse: '18.8793',
                    rmse: '4.3450',
                    mae: '3.8380',
                    r2: '0.6756',
                },
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            neuralNetworkPage = new NeuralNetworkPage(page, DATASET_FILE);
            await neuralNetworkPage.navigateToRegressionPage();
            await neuralNetworkPage.configureDataset();
            await neuralNetworkPage.setBasicConfiguration();

            // Speed up tests
            await neuralNetworkPage.setMaxIterations(25);
        });

        test('should successfully train model by default', async () => {
            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await neuralNetworkPage.verifyTrainingResults(
                defaultExpected.trainLoss,
                defaultExpected.testLoss,
            );
            await neuralNetworkPage.verifyRegressionMetrics(defaultExpected.metrics);
        });

        test('should successfully train model using transformation', async ({ page }) => {
            await neuralNetworkPage.setNormalization('Z-Score');
            await neuralNetworkPage.addTransformation('Polynomial', { degree: 3 });

            await expect(page.getByText('Output features: 2')).toBeVisible();

            await neuralNetworkPage.setLearningRate(0.1);

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const transformationExpected = EXPECTED_RESULTS.withTransformation;
            await neuralNetworkPage.verifyTrainingResults(
                transformationExpected.trainLoss,
                transformationExpected.testLoss,
            );
            await neuralNetworkPage.verifyRegressionMetrics(transformationExpected.metrics);
        });

        test('should successfully train model using several layers RELU', async () => {
            await neuralNetworkPage.addLayer(3, 'ReLU');
            await neuralNetworkPage.addLayer(2, 'ReLU');
            await neuralNetworkPage.addLayer(1, 'Linear');
            await neuralNetworkPage.removeLayer(0); // Remove default first layer

            await neuralNetworkPage.setNormalization('Z-Score');

            await neuralNetworkPage.setMaxIterations(100);

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const layersExpected = EXPECTED_RESULTS.withLayers;
            await neuralNetworkPage.verifyTrainingResults(
                layersExpected.trainLoss,
                layersExpected.testLoss,
            );
            await neuralNetworkPage.verifyRegressionMetrics(layersExpected.metrics);
        });
    });

    test.describe('Binary Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 87.50%',
                testAccuracy: 'Test Accuracy: 100.00%',
                confusionMatrixMetrics: {
                    Accuracy: '87.5%',
                    MCC: '0.741',
                    "Cohen's Kappa": '0.731',
                    Precision: '92.3%',
                    Recall: '75.0%',
                    F1: '82.8%',
                },
                rocAuc: { auc: '0.945' },
            },
            withLayers: {
                trainAccuracy: 'Train Accuracy: 60.00%',
                testAccuracy: 'Test Accuracy: 60.00%',
                confusionMatrixMetrics: {
                    Accuracy: '60.0%',
                    MCC: '0.000',
                    "Cohen's Kappa": '0.000',
                    Precision: '0.0%',
                    Recall: '0.0%',
                    F1: '0.0%',
                },
                rocAuc: { auc: '0.625' },
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            neuralNetworkPage = new NeuralNetworkPage(page, DATASET_FILE);
            await neuralNetworkPage.navigateToClassificationPage();
            await neuralNetworkPage.configureDataset();
            await neuralNetworkPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await neuralNetworkPage.configureLayer(0, { units: 1, activation: 'Sigmoid' });

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await neuralNetworkPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await neuralNetworkPage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
            await neuralNetworkPage.verifyROCAUC(defaultExpected.rocAuc);
        });

        test('should successfully train model using several layers RELU', async () => {
            await neuralNetworkPage.addLayer(3, 'ReLU');
            await neuralNetworkPage.addLayer(2, 'ReLU');
            await neuralNetworkPage.addLayer(1, 'Sigmoid');
            await neuralNetworkPage.removeLayer(0); // Remove default first layer

            await neuralNetworkPage.setNormalization('Z-Score');

            await neuralNetworkPage.setOptimizer('Adam');
            await neuralNetworkPage.setLearningRate(0.1);

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const layersExpected = EXPECTED_RESULTS.withLayers;
            await neuralNetworkPage.verifyTrainingResults(
                layersExpected.trainAccuracy,
                layersExpected.testAccuracy,
            );
            await neuralNetworkPage.verifyConfusionMatrixMetrics(
                layersExpected.confusionMatrixMetrics,
            );
            await neuralNetworkPage.verifyROCAUC(layersExpected.rocAuc);
        });
    });

    test.describe('Multiclass Classification (Softmax)', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
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
                multiclassROCAUC: { macroAuc: '1.000', weightedAuc: '1.000' },
            },
            withMomentum: {
                trainAccuracy: 'Train Accuracy: 68.75%',
                testAccuracy: 'Test Accuracy: 33.33%',
                confusionMatrixMetrics: {
                    Accuracy: '68.8%',
                    MCC: '0.599',
                    "Cohen's Kappa": '0.505',
                    'Macro Precision': '51.5%',
                    'Macro Recall': '62.7%',
                    'Macro F1': '54.8%',
                    'Weighted Precision': '68.8%',
                    'Weighted Recall': '68.8%',
                    'Weighted F1': '59.7%',
                },
                multiclassROCAUC: { macroAuc: '0.822', weightedAuc: '0.824' },
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            neuralNetworkPage = new NeuralNetworkPage(page, DATASET_FILE);
            await neuralNetworkPage.navigateToClassificationPage();
            await neuralNetworkPage.configureDataset();
            await neuralNetworkPage.setBasicConfiguration();

            await neuralNetworkPage.setMaxIterations(25); // Speed up tests
        });

        test('should successfully train model by default', async () => {
            await neuralNetworkPage.configureLayer(0, { units: 3, activation: 'Softmax' });

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await neuralNetworkPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await neuralNetworkPage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
            await neuralNetworkPage.verifyMulticlassROCAUC(defaultExpected.multiclassROCAUC);
        });

        test('should successfully train model using Momentum', async () => {
            await neuralNetworkPage.addLayer(3, 'ReLU');
            await neuralNetworkPage.addLayer(2, 'ReLU');
            await neuralNetworkPage.addLayer(3, 'Softmax');
            await neuralNetworkPage.removeLayer(0); // Remove default first layer

            await neuralNetworkPage.setNormalization('Z-Score');

            await neuralNetworkPage.setOptimizer('Momentum');
            await neuralNetworkPage.setLearningRate(0.1);

            await neuralNetworkPage.setLossFunction('Categorical cross-entropy');

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            const momentumExpected = EXPECTED_RESULTS.withMomentum;
            await neuralNetworkPage.verifyTrainingResults(
                momentumExpected.trainAccuracy,
                momentumExpected.testAccuracy,
            );
            await neuralNetworkPage.verifyConfusionMatrixMetrics(
                momentumExpected.confusionMatrixMetrics,
            );
            await neuralNetworkPage.verifyMulticlassROCAUC(momentumExpected.multiclassROCAUC);
        });
    });
});
