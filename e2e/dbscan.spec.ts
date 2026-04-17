import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { DBSCANPage } from './pages/DBSCANPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('DBSCAN Training', () => {
    let dbscanPage: DBSCANPage;

    const DATASET_FILE = path.resolve(__dirname, './data/clustering_test_data.csv');

    test.describe('Clustering', () => {
        const EXPECTED_RESULTS = {
            default: {
                trainSilhouette: 'Train Silhouette: 0.8167',
                testSilhouette: 'Test Silhouette: 0.9552',
            },
            withEps2Min3: {
                trainSilhouette: 'Train Silhouette: 0.8799',
                testSilhouette: 'Test Silhouette: 0.8498',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            dbscanPage = new DBSCANPage(page, DATASET_FILE);
            await dbscanPage.navigateToClusteringPage();
            await dbscanPage.configureDataset();
            await dbscanPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await dbscanPage.startTraining();
            await dbscanPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await dbscanPage.verifyTrainingResults(
                defaultExpected.trainSilhouette,
                defaultExpected.testSilhouette,
            );
        });

        test('should successfully train model with eps=2 and minPoints=3', async () => {
            await dbscanPage.setEpsilon(2);
            await dbscanPage.setMinPoints(3);

            await dbscanPage.startTraining();
            await dbscanPage.waitForTrainingCompletion();

            const eps2Expected = EXPECTED_RESULTS.withEps2Min3;
            await dbscanPage.verifyTrainingResults(
                eps2Expected.trainSilhouette,
                eps2Expected.testSilhouette,
            );
        });
    });

    test.describe('Anomaly Detection', () => {
        const EXPECTED_RESULTS = {
            default: {
                trainAnomalies: 'Train Anomalies: 23.08%',
                testAnomalies: 'Test Anomalies: 46.15%',
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            dbscanPage = new DBSCANPage(page, DATASET_FILE);
            await dbscanPage.navigateToAnomalyPage();
            await dbscanPage.configureDataset();
            await dbscanPage.setBasicConfiguration();
        });

        test('should successfully detect anomalies by default', async () => {
            await dbscanPage.startTraining();
            await dbscanPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await dbscanPage.verifyTrainingResults(
                defaultExpected.trainAnomalies,
                defaultExpected.testAnomalies,
            );
        });
    });
});
