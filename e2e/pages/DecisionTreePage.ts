import { expect } from '@playwright/test';
import { LinearRegressionPage } from './LinearRegressionPage';

type CriterionType =
    | 'Gini'
    | 'Entropy'
    | 'MSE (Mean Squared Error)'
    | 'MAE (Mean Absolute Error)'
    | 'Huber';
type ModelVariantType = 'Single Decision Tree' | 'Bagging' | 'Random Forest' | 'Extra Trees';

export class DecisionTreePage extends LinearRegressionPage {
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
        await this.page.getByRole('option', { name: 'Decision Tree', exact: true }).click();
    }

    async setModelVariant(variant: ModelVariantType): Promise<void> {
        await this.page.getByRole('radio', { name: variant }).click();
    }

    async setCriterion(criterion: CriterionType, huberDelta?: number): Promise<void> {
        await this.page.getByTestId('criterion-select').click();
        await this.page.getByRole('option', { name: criterion, exact: true }).click();

        if (criterion === 'Huber' && huberDelta !== undefined) {
            await this.page.getByTestId('huber-delta-input').fill(huberDelta.toString());
        }
    }

    async setMaxDepth(maxDepth: number): Promise<void> {
        await this.page.getByTestId('max-depth-input').fill(maxDepth.toString());
    }

    async setMinSamplesSplit(minSamplesSplit: number): Promise<void> {
        await this.page.getByTestId('min-samples-split-input').fill(minSamplesSplit.toString());
    }

    async setMinSamplesLeaf(minSamplesLeaf: number): Promise<void> {
        await this.page.getByTestId('min-samples-leaf-input').fill(minSamplesLeaf.toString());
    }

    async setMaxFeatures(maxFeatures: number): Promise<void> {
        await this.page.getByTestId('max-features-input').fill(maxFeatures.toString());
    }

    async setEstimators(estimators: number): Promise<void> {
        await this.page.getByTestId('estimators-input').fill(estimators.toString());
    }

    async setNumRandomThresholds(numRandomThresholds: number): Promise<void> {
        await this.page.getByTestId('random-thresholds-input').fill(numRandomThresholds.toString());
    }
}
