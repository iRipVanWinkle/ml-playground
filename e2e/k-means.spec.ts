import { test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { KMeansPage } from './pages/KMeansPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('K-Means Training', () => {
    let kMeansPage: KMeansPage;

    const DATASET_FILE = path.resolve(__dirname, './data/clustering_test_data.csv');
    const EXPECTED_RESULTS = {
        default: {
            trainSilhouette: 'Train Silhouette: 0.7507',
            testSilhouette: 'Test Silhouette: 0.7153',
            clusterMetrics: [
                {
                    'Silhouette Score': '0.7606',
                    Radius: '8.0522',
                    Cohesion: '1.1886',
                    Separation: '7.2549',
                },
                {
                    'Silhouette Score': '0.6993',
                    Radius: '9.0177',
                    Cohesion: '1.6020',
                    Separation: '7.9869',
                },
                {
                    'Silhouette Score': '0.7922',
                    Radius: '6.2635',
                    Cohesion: '0.9889',
                    Separation: '7.2398',
                },
            ],
        },
    } as const;

    test.beforeEach(async ({ page }) => {
        kMeansPage = new KMeansPage(page, DATASET_FILE);
        await kMeansPage.navigateToPage();
        await kMeansPage.configureDataset();
        await kMeansPage.setBasicConfiguration();
    });

    test('should successfully train model by default', async () => {
        await kMeansPage.startTraining();
        await kMeansPage.waitForTrainingCompletion();

        const defaultExpected = EXPECTED_RESULTS.default;
        await kMeansPage.verifyTrainingResults(
            defaultExpected.trainSilhouette,
            defaultExpected.testSilhouette,
        );
        await kMeansPage.verifyClusterMetrics(defaultExpected.clusterMetrics);
    });
});
