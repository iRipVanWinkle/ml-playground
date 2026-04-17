import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { IsolationForestPage } from './pages/IsolationForestPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Isolation Forest Anomaly Detection', () => {
    let isolationForestPage: IsolationForestPage;

    const DATASET_FILE = path.resolve(__dirname, './data/clustering_test_data.csv');
    const EXPECTED_RESULTS = {
        default: {
            trainAnomalies: 'Train Anomalies: 11.54%',
            testAnomalies: 'Test Anomalies: 7.69%',
        },
        withContamination02: {
            trainAnomalies: 'Train Anomalies: 21.15%',
            testAnomalies: 'Test Anomalies: 38.46%',
        },
        withEstimators50Bootstrap: {
            trainAnomalies: 'Train Anomalies: 11.54%',
            testAnomalies: 'Test Anomalies: 15.38%',
        },
    } as const;

    test.beforeEach(async ({ page }) => {
        isolationForestPage = new IsolationForestPage(page, DATASET_FILE);
        await isolationForestPage.navigateToPage();
        await isolationForestPage.configureDataset();
        await isolationForestPage.setBasicConfiguration();
    });

    test('should successfully detect anomalies by default', async () => {
        await isolationForestPage.setEstimators(5);

        await isolationForestPage.startTraining();
        await isolationForestPage.waitForTrainingCompletion();

        const defaultExpected = EXPECTED_RESULTS.default;
        await isolationForestPage.verifyTrainingResults(
            defaultExpected.trainAnomalies,
            defaultExpected.testAnomalies,
        );
    });

    test('should detect more anomalies with higher contamination', async () => {
        await isolationForestPage.setEstimators(5);
        await isolationForestPage.setContamination(0.2);

        await isolationForestPage.startTraining();
        await isolationForestPage.waitForTrainingCompletion();

        const contaminationExpected = EXPECTED_RESULTS.withContamination02;
        await isolationForestPage.verifyTrainingResults(
            contaminationExpected.trainAnomalies,
            contaminationExpected.testAnomalies,
        );
    });

    test('should successfully detect anomalies with 50 estimators and bootstrap', async () => {
        await isolationForestPage.setEstimators(5);
        await isolationForestPage.setBootstrap(true);

        await isolationForestPage.startTraining();
        await isolationForestPage.waitForTrainingCompletion();

        const bootstrapExpected = EXPECTED_RESULTS.withEstimators50Bootstrap;
        await isolationForestPage.verifyTrainingResults(
            bootstrapExpected.trainAnomalies,
            bootstrapExpected.testAnomalies,
        );
    });
});
