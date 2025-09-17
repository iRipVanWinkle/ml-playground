import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

export class LogisticRegressionPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.page.getByRole('tab', { name: 'Classification' }).click();
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setClassificationType(type: 'Binary' | 'One-vs-Rest' | 'Softmax'): Promise<void> {
        if (type === 'Binary') {
            await this.page.getByTestId('classification-type-binary').click();
        } else if (type === 'One-vs-Rest') {
            await this.page.getByTestId('classification-type-ovr').click();
        } else {
            await this.page.getByTestId('classification-type-softmax').click();
        }
    }
}
