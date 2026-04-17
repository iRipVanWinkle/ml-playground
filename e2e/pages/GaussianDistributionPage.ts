import { AnomalyDetectionPage } from './AnomalyDetectionPage';

type VariantType = 'Diagonal' | 'Full';

export class GaussianDistributionPage extends AnomalyDetectionPage {
    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'Gaussian Distribution', exact: true }).click();
    }

    async setVariant(variant: VariantType): Promise<void> {
        await this.page.getByTestId('gaussian-variant-select').click();
        await this.page.getByRole('option', { name: variant, exact: true }).click();
    }

    async setThreshold(threshold: number): Promise<void> {
        await this.page.getByTestId('gaussian-threshold-input').fill(threshold.toString());
    }

    async setVarianceSmoothing(smoothing: number): Promise<void> {
        await this.page.getByTestId('gaussian-smoothing-input').fill(smoothing.toString());
    }
}
