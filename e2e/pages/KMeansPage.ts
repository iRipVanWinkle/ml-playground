import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type CentroidInitType = 'Random' | 'K-Means++';
type DistanceType = 'Euclidean' | 'Manhattan' | 'Cosine';

export class KMeansPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Clustering');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'K-Means', exact: true }).click();
    }

    async setNumClusters(k: number): Promise<void> {
        await this.page.getByTestId('num-clusters-input').fill(k.toString());
    }

    async setMaxIterations(maxIterations: number): Promise<void> {
        await this.page.getByTestId('max-iterations-input').fill(maxIterations.toString());
    }

    async setCentroidInitialization(type: CentroidInitType): Promise<void> {
        await this.page.getByTestId('centroid-initialization-select').click();
        await this.page.getByRole('option', { name: type, exact: true }).click();
    }

    async setDistance(distance: DistanceType): Promise<void> {
        await this.page.getByTestId('distance-select').click();
        await this.page.getByRole('option', { name: distance, exact: true }).click();
    }

    async verifyClusterMetrics(expected: ReadonlyArray<Record<string, string>>): Promise<void> {
        await this.navigateToVisualizationTab('Metrics');

        for (let i = 0; i < expected.length; i++) {
            for (const [label, value] of Object.entries(expected[i])) {
                await expect(this.page.getByTestId(`cluster-${i}-${label}-value`)).toHaveText(
                    value,
                );
            }
        }
    }
}
