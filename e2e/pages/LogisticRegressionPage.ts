import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

export class LogisticRegressionPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.page.getByRole('tab', { name: 'Classification' }).click();
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }
}
