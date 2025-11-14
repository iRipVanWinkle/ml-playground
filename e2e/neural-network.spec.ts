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
            },
            withTransformation: {
                trainLoss: 'Train Loss: 1.1848',
                testLoss: 'Test Loss: 19.9749',
            },
            withLayers: {
                trainLoss: 'Train Loss: 18.8793',
                testLoss: 'Test Loss: 1.1578',
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

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainLoss,
                EXPECTED_RESULTS.default.testLoss,
            );
        });

        test('should successfully train model using transformation', async ({ page }) => {
            await neuralNetworkPage.setNormalization('Z-Score');
            await neuralNetworkPage.addTransformation('Polynomial', { degree: 3 });

            await expect(page.getByText('Output features: 2')).toBeVisible();

            await neuralNetworkPage.setLearningRate(0.1);

            await neuralNetworkPage.startTraining();
            await neuralNetworkPage.waitForTrainingCompletion();

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.withTransformation.trainLoss,
                EXPECTED_RESULTS.withTransformation.testLoss,
            );
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

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.withLayers.trainLoss,
                EXPECTED_RESULTS.withLayers.testLoss,
            );
        });
    });

    test.describe('Binary Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 87.50%',
                testAccuracy: 'Test Accuracy: 100.00%',
            },
            withLayers: {
                trainAccuracy: 'Train Accuracy: 60.00%',
                testAccuracy: 'Test Accuracy: 60.00%',
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

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
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

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.withLayers.trainAccuracy,
                EXPECTED_RESULTS.withLayers.testAccuracy,
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
                trainAccuracy: 'Train Accuracy: 100.00%',
                testAccuracy: 'Test Accuracy: 100.00%',
            },
            withTransformation: {
                trainAccuracy: 'Train Accuracy: 68.75%',
                testAccuracy: 'Test Accuracy: 33.33%',
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

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.default.trainAccuracy,
                EXPECTED_RESULTS.default.testAccuracy,
            );
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

            await neuralNetworkPage.verifyTrainingResults(
                EXPECTED_RESULTS.withTransformation.trainAccuracy,
                EXPECTED_RESULTS.withTransformation.testAccuracy,
            );
        });
    });
});
