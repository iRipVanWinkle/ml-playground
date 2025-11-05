import { expect, Page } from '@playwright/test';

type NormalizationType = 'Z-Score' | 'None';
type TransformationType = 'Sinusoid' | 'Polynomial';
type LossFunctionType =
    | 'MSE (Mean Squared Error)'
    | 'MAE (Mean Absolute Error)'
    | 'Huber'
    | 'Binary cross-entropy'
    | 'Categorical cross-entropy';
type OptimizerType = 'Batch Gradient Descent' | 'Stochastic Gradient Descent' | 'Momentum' | 'Adam';
type RegularizationType = 'L2 (Ridge)' | 'None';
type WeightInitializationType = 'Constant' | 'Zeros';

export class LinearRegressionPage {
    protected page: Page;
    private dataset: string;

    private consoleMessages: string[] = [];

    constructor(page: Page, dataset: string) {
        this.page = page;
        this.dataset = dataset;

        this.page.on('console', (msg) => {
            if (msg.type() === 'error') {
                this.consoleMessages.push(msg.text());
            }
        });
    }

    async navigateToPage(): Promise<void> {
        await this.page.goto('/');
        await expect(this.page).toHaveTitle('Machine Learning Playground');
    }

    async navigateToTab(tab: 'Regression' | 'Classification'): Promise<void> {
        await this.page.getByTestId('task-switcher-list').getByRole('tab', { name: tab }).click();
    }

    async setBasicConfiguration(): Promise<void> {
        await this.page.getByTestId('tensorflow-backend-select').click();
        await this.page.getByRole('option', { name: 'CPU', exact: true }).click();
    }

    async configureDataset(dataset?: string): Promise<void> {
        await this.page.getByTestId('dataset-select').click();
        await this.page.getByRole('option', { name: 'Custom Dataset', exact: true }).click();
        await this.page.getByTestId('custom-dataset-input').setInputFiles(dataset ?? this.dataset);
    }

    async setNormalization(type: NormalizationType): Promise<void> {
        if (type) {
            await this.page.getByTestId('normalization-select').click();
            await this.page.getByRole('option', { name: type, exact: true }).click();
        }
    }

    async addTransformation(
        type: TransformationType,
        options?: { degree?: number },
    ): Promise<void> {
        await this.page.getByTestId('add-transformation-button').click();

        const transformationContainers = this.page.getByTestId('transformation-container');
        const newContainer = transformationContainers.last();

        const typeSelect = newContainer.getByTestId('transformation-type-select');
        await expect(typeSelect).toBeVisible();

        await typeSelect.click();
        await this.page.getByRole('option', { name: type, exact: true }).click();

        if (options?.degree) {
            await newContainer.getByTestId('degree-input').fill(options.degree.toString());
        }
    }

    async removeTransformation(index: number): Promise<void> {
        const transformationContainers = this.page.getByTestId('transformation-container');
        const targetContainer = transformationContainers.nth(index);
        await targetContainer.getByTestId('remove-transformation-button').click();
    }

    async configureTransformation(
        index: number,
        options: { type?: TransformationType; degree?: number },
    ): Promise<void> {
        const transformationContainers = this.page.getByTestId('transformation-container');
        const targetContainer = transformationContainers.nth(index);

        if (options.type) {
            await targetContainer.getByTestId('transformation-type-select').click();
            await this.page.getByRole('option', { name: options.type, exact: true }).click();
        }

        if (options.degree) {
            await targetContainer.getByTestId('degree-input').fill(options.degree.toString());
        }
    }

    async setLossFunction(type: LossFunctionType, huberDelta?: number): Promise<void> {
        await this.page.getByTestId('loss-function-select').click();
        await this.page.getByRole('option', { name: type, exact: true }).click();

        if (type === 'Huber' && huberDelta) {
            await this.page.getByTestId('huber-delta-input').fill(huberDelta.toString());
        }
    }

    async setOptimizer(type: OptimizerType, options: { batchSize?: number } = {}): Promise<void> {
        await this.page.getByTestId('optimizer-select').click();
        await this.page.getByRole('option', { name: type, exact: true }).click();

        if (options.batchSize) {
            await this.page.getByTestId('batch-size-input').fill(options.batchSize.toString());
        }
    }

    async setLearningRate(
        learningRate: number,
        schedulerConfig?: { s: number; p: number },
    ): Promise<void> {
        await this.page.getByTestId('learning-rate-input').fill(learningRate.toString());

        if (schedulerConfig) {
            await this.page.getByTestId('scheduler-checkbox').click();
            await this.page.getByTestId('decay-offset-input').fill(schedulerConfig.s.toString());
            await this.page.getByTestId('decay-power-input').fill(schedulerConfig.p.toString());
        }
    }

    async setMaxIterations(maxIterations: number): Promise<void> {
        await this.page.getByTestId('max-iterations-input').fill(maxIterations.toString());
    }

    async setRegularization(
        type: RegularizationType,
        options: { lambda?: number } = {},
    ): Promise<void> {
        await this.page.getByTestId('regularization-select').click();
        await this.page.getByRole('option', { name: type, exact: true }).click();

        if (options.lambda) {
            await this.page.getByTestId('lambda-input').fill(options.lambda.toString());
        }
    }

    async setWeightInitialization(
        type: WeightInitializationType,
        options: { constant?: string } = {},
    ): Promise<void> {
        await this.page.getByTestId('theta-initialization-select').click();
        await this.page.getByRole('option', { name: type, exact: true }).click();

        if (options.constant) {
            await this.page.getByTestId('constant-value-input').fill(options.constant);
        }
    }

    async startTraining(): Promise<void> {
        await this.page.getByTestId('start-training').click();
    }

    async waitForTrainingCompletion(): Promise<void> {
        await expect(
            this.page.locator('[data-sonner-toast]').filter({ hasText: 'Training finished' }),
        ).toBeVisible({ timeout: 15000 });
    }

    async verifyTrainingResults(
        expectedTrainLoss: string,
        expectedTestLoss: string,
    ): Promise<void> {
        await expect(this.page.getByText(expectedTrainLoss)).toBeVisible();
        await expect(this.page.getByText(expectedTestLoss)).toBeVisible();

        expect(this.consoleMessages).toHaveLength(0);
    }
}
