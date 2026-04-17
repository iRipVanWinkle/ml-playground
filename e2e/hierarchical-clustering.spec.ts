import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { HierarchicalClusteringPage } from './pages/HierarchicalClusteringPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Hierarchical Clustering Training', () => {
    let hierarchicalPage: HierarchicalClusteringPage;

    const DATASET_FILE = path.resolve(__dirname, './data/clustering_test_data.csv');
    const EXPECTED_RESULTS = {
        divisiveDefault: {
            clusters: 'Clusters: 3',
        },
        agglomerativeWard: {
            clusters: 'Clusters: 3',
        },
        agglomerativeComplete: {
            clusters: 'Clusters: 3',
        },
    } as const;

    test.beforeEach(async ({ page }) => {
        hierarchicalPage = new HierarchicalClusteringPage(page, DATASET_FILE);
        await hierarchicalPage.navigateToPage();
        await hierarchicalPage.configureDataset();
        await hierarchicalPage.setBasicConfiguration();
    });

    test('should successfully train model with Divisive method by default', async ({ page }) => {
        await hierarchicalPage.startTraining();
        await hierarchicalPage.waitForTrainingCompletion();

        await expect(page.getByText(EXPECTED_RESULTS.divisiveDefault.clusters)).toBeVisible();
    });

    test('should successfully train model with Agglomerative Ward linkage', async ({ page }) => {
        await hierarchicalPage.setMethod('Agglomerative');
        await hierarchicalPage.setLinkage('Ward');

        await hierarchicalPage.startTraining();
        await hierarchicalPage.waitForTrainingCompletion();

        await expect(page.getByText(EXPECTED_RESULTS.agglomerativeWard.clusters)).toBeVisible();
    });

    test('should successfully train model with Agglomerative Complete linkage', async ({
        page,
    }) => {
        await hierarchicalPage.setMethod('Agglomerative');
        await hierarchicalPage.setLinkage('Complete');

        await hierarchicalPage.startTraining();
        await hierarchicalPage.waitForTrainingCompletion();

        await expect(page.getByText(EXPECTED_RESULTS.agglomerativeComplete.clusters)).toBeVisible();
    });
});
