import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type KNNWeightsType = 'Uniform' | 'Distance';
type KNNDistanceType = 'Euclidean' | 'Manhattan' | 'Cosine';

export class KNNPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        throw new Error('Use navigateToRegressionPage or navigateToClassificationPage instead');
    }

    async navigateToRegressionPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Regression');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async navigateToClassificationPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Classification');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'K-Nearest Neighbors', exact: true }).click();
    }

    async setK(k: number): Promise<void> {
        await this.page.getByTestId('knn-k-input').fill(k.toString());
    }

    async setWeights(weights: KNNWeightsType): Promise<void> {
        await this.page.getByTestId('knn-weights-select').click();
        await this.page.getByRole('option', { name: weights, exact: true }).click();
    }

    async setDistance(distance: KNNDistanceType): Promise<void> {
        await this.page.getByTestId('knn-distance-select').click();
        await this.page.getByRole('option', { name: distance, exact: true }).click();
    }
}
