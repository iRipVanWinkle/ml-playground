import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { GaussianDistributionPage } from './pages/GaussianDistributionPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Gaussian Distribution Anomaly Detection', () => {
    let gaussianPage: GaussianDistributionPage;

    const DATASET_FILE = path.resolve(__dirname, './data/clustering_test_data.csv');
    const EXPECTED_RESULTS = {
        default: {
            trainAnomalies: 'Train Anomalies: 7.69%',
            testAnomalies: 'Test Anomalies: 7.69%',
        },
        withFullVariant: {
            trainAnomalies: 'Train Anomalies: 7.69%',
            testAnomalies: 'Test Anomalies: 7.69%',
        },
        withThreshold001: {
            trainAnomalies: 'Train Anomalies: 98.08%',
            testAnomalies: 'Test Anomalies: 100.00%',
        },
    } as const;

    test.beforeEach(async ({ page }) => {
        gaussianPage = new GaussianDistributionPage(page, DATASET_FILE);
        await gaussianPage.navigateToPage();
        await gaussianPage.configureDataset();
        await gaussianPage.setBasicConfiguration();
    });

    test('should detect no anomalies by default with Diagonal variant', async () => {
        await gaussianPage.startTraining();
        await gaussianPage.waitForTrainingCompletion();

        const defaultExpected = EXPECTED_RESULTS.default;
        await gaussianPage.verifyTrainingResults(
            defaultExpected.trainAnomalies,
            defaultExpected.testAnomalies,
        );
    });

    test('should detect no anomalies with Full variant', async () => {
        await gaussianPage.setVariant('Full');

        await gaussianPage.startTraining();
        await gaussianPage.waitForTrainingCompletion();

        const fullExpected = EXPECTED_RESULTS.withFullVariant;
        await gaussianPage.verifyTrainingResults(
            fullExpected.trainAnomalies,
            fullExpected.testAnomalies,
        );
    });

    test('should detect anomalies with higher threshold', async () => {
        await gaussianPage.setThreshold(0.01);

        await gaussianPage.startTraining();
        await gaussianPage.waitForTrainingCompletion();

        const thresholdExpected = EXPECTED_RESULTS.withThreshold001;
        await gaussianPage.verifyTrainingResults(
            thresholdExpected.trainAnomalies,
            thresholdExpected.testAnomalies,
        );
    });
});
