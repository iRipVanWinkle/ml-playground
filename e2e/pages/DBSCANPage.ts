import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type DistanceType = 'Euclidean' | 'Manhattan' | 'Cosine';

export class DBSCANPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        throw new Error('Use navigateToClusteringPage or navigateToAnomalyPage instead');
    }

    async navigateToClusteringPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Clustering');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async navigateToAnomalyPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Anomaly');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'DBSCAN', exact: true }).click();
    }

    async setEpsilon(epsilon: number): Promise<void> {
        await this.page.getByTestId('epsilon-input').fill(epsilon.toString());
    }

    async setMinPoints(minPoints: number): Promise<void> {
        await this.page.getByTestId('min-points-input').fill(minPoints.toString());
    }

    async setDistance(distance: DistanceType): Promise<void> {
        await this.page.getByTestId('distance-select').click();
        await this.page.getByRole('option', { name: distance, exact: true }).click();
    }
}
