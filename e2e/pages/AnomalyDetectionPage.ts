import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

export class AnomalyDetectionPage extends LinearRegressionPage {
    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await this.navigateToTab('Anomaly');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }
}
