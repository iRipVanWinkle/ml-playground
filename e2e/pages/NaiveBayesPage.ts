import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type NaiveBayesVariantType = 'Gaussian' | 'Quadratic';

export class NaiveBayesPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Classification');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'Naive Bayes', exact: true }).click();
    }

    async setVariant(variant: NaiveBayesVariantType): Promise<void> {
        await this.page.getByTestId('naive-bayes-variant-select').click();
        await this.page.getByRole('option', { name: variant, exact: true }).click();
    }
}
