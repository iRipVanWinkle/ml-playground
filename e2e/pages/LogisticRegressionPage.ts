import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

export class LogisticRegressionPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Classification');
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

    async verifyMulticlassLearnedParameters(
        expected: Array<{ bias: string; weights: Record<string, string> }>,
    ): Promise<void> {
        const biasElements = this.page.getByTestId('param-bias-value');
        await expect(biasElements).toHaveCount(expected.length);

        for (let i = 0; i < expected.length; i++) {
            await expect(biasElements.nth(i)).toHaveText(expected[i].bias);
            for (const [feature, value] of Object.entries(expected[i].weights)) {
                await expect(
                    this.page.getByTestId(`param-weight-${feature}-value`).nth(i),
                ).toHaveText(value);
            }
        }
    }
}
