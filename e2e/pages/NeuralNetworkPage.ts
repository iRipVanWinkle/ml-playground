import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type ActivationType = 'Linear' | 'ReLU' | 'Sigmoid' | 'Tanh' | 'Softmax';

export class NeuralNetworkPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        throw new Error('Use navigateToRegressionPage or navigateToClassificationPage instead');
    }

    async navigateToRegressionPage(): Promise<void> {
        await this.page.goto('/');
        await this.page.getByRole('tab', { name: 'Regression' }).click();
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async navigateToClassificationPage(): Promise<void> {
        await this.page.goto('/');
        await this.page.getByRole('tab', { name: 'Classification' }).click();
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'Neural Networks', exact: true }).click();
    }

    // Add method for manipulation of layers
    async addLayer(units: number, activation: ActivationType): Promise<void> {
        await this.page.getByTestId('add-layer-button').click();

        const layerItems = this.page.getByTestId('layer-item');
        const newLayer = layerItems.last();

        const unitsInput = newLayer.getByTestId('units-input');
        await expect(unitsInput).toBeVisible();
        await unitsInput.fill(units.toString());

        const activationSelect = newLayer.getByTestId('activation-select');
        await activationSelect.click();
        await this.page.getByRole('option', { name: activation, exact: true }).click();
    }

    async removeLayer(index: number): Promise<void> {
        const layerItems = this.page.getByTestId('layer-item');
        const layerToRemove = layerItems.nth(index);
        const removeButton = layerToRemove.getByTestId('remove-layer-button');
        await removeButton.click();
    }

    async configureLayer(
        index: number,
        options: { units?: number; activation?: ActivationType },
    ): Promise<void> {
        const layerItems = this.page.getByTestId('layer-item');
        const targetLayer = layerItems.nth(index);

        if (options.units) {
            const unitsInput = targetLayer.getByTestId('units-input');
            await expect(unitsInput).toBeVisible();
            await unitsInput.fill(options.units.toString());
        }

        if (options.activation) {
            const activationSelect = targetLayer.getByTestId('activation-select');
            await activationSelect.click();
            await this.page.getByRole('option', { name: options.activation, exact: true }).click();
        }
    }
}
