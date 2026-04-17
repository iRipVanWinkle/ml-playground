import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type MethodType = 'Divisive' | 'Agglomerative';
type LinkageType = 'Ward' | 'Complete' | 'Average' | 'Single';
type DistanceType = 'Euclidean' | 'Manhattan' | 'Cosine';

export class HierarchicalClusteringPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Clustering');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page
            .getByRole('option', { name: 'Hierarchical Clustering', exact: true })
            .click();
    }

    async setMethod(method: MethodType): Promise<void> {
        await this.page.getByRole('radio', { name: method }).click();
    }

    async setNumClusters(k: number): Promise<void> {
        await this.page.getByTestId('num-clusters-input').fill(k.toString());
    }

    async setDistance(distance: DistanceType): Promise<void> {
        await this.page.getByTestId('distance-select').click();
        await this.page.getByRole('option', { name: distance, exact: true }).click();
    }

    async setLinkage(linkage: LinkageType): Promise<void> {
        await this.page.getByTestId('linkage-select').click();
        await this.page.getByRole('option', { name: linkage, exact: true }).click();
    }
}
