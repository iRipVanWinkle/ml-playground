import { AnomalyDetectionPage } from './AnomalyDetectionPage';

export class IsolationForestPage extends AnomalyDetectionPage {
    async setBasicConfiguration(): Promise<void> {
        await super.setBasicConfiguration();
        await this.page.getByTestId('model-type-select').click();
        await this.page.getByRole('option', { name: 'Isolation Forest', exact: true }).click();
    }

    async setEstimators(estimators: number): Promise<void> {
        await this.page.getByTestId('if-estimators-input').fill(estimators.toString());
    }

    async setMaxSamples(maxSamples: number): Promise<void> {
        await this.page.getByTestId('if-max-samples-input').fill(maxSamples.toString());
    }

    async setContamination(contamination: number): Promise<void> {
        await this.page.getByTestId('if-contamination-input').fill(contamination.toString());
    }

    async setBootstrap(enabled: boolean): Promise<void> {
        const switchEl = this.page.getByTestId('if-bootstrap-switch');
        const isChecked = (await switchEl.getAttribute('aria-checked')) === 'true';
        if (isChecked !== enabled) {
            await switchEl.click();
        }
    }
}
