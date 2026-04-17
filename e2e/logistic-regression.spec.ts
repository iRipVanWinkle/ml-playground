import { expect, test } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import { LogisticRegressionPage } from './pages/LogisticRegressionPage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Logistic Regression Training', () => {
    let logisticRegressionPage: LogisticRegressionPage;

    test.describe('Binary Classification', () => {
        const DATASET_FILE = path.resolve(__dirname, './data/logistic_regression_test_data.csv');
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 90.00%',
                testAccuracy: 'Test Accuracy: 100.00%',
                confusionMatrixMetrics: {
                    Accuracy: '90.0%',
                    MCC: '0.792',
                    "Cohen's Kappa": '0.787',
                    Precision: '92.9%',
                    Recall: '81.3%',
                    F1: '86.7%',
                },
                rocAuc: { auc: '0.979' },
                learnedParameters: {
                    bias: '0.0554',
                    weights: { x1: '-0.4084', x2: '+0.3865' },
                },
            },
            withTransformation: {
                trainAccuracy: 'Train Accuracy: 92.50%',
                testAccuracy: 'Test Accuracy: 70.00%',
                confusionMatrixMetrics: {
                    Accuracy: '92.5%',
                    MCC: '0.846',
                    "Cohen's Kappa": '0.845',
                    Precision: '88.2%',
                    Recall: '93.8%',
                    F1: '90.9%',
                },
                rocAuc: { auc: '0.992' },
                learnedParameters: {
                    bias: '0.0882',
                    weights: {
                        x1: '-0.2776',
                        x2: '+0.1196',
                        'x2^2': '-0.1339',
                        'x1*x2': '+0.0285',
                        'x1^2': '-0.0562',
                        'x2^3': '+0.1344',
                        'x1*x2^2': '-0.1988',
                        'x1^2*x2': '-0.0286',
                        'x1^3': '-0.2428',
                    },
                },
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            logisticRegressionPage = new LogisticRegressionPage(page, DATASET_FILE);
            await logisticRegressionPage.navigateToPage();
            await logisticRegressionPage.configureDataset();
            await logisticRegressionPage.setBasicConfiguration();
        });

        test('should successfully train model by default', async () => {
            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await logisticRegressionPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await logisticRegressionPage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
            await logisticRegressionPage.verifyROCAUC(defaultExpected.rocAuc);
            await logisticRegressionPage.verifyLearnedParameters(defaultExpected.learnedParameters);
        });

        test('should successfully train model using transformation', async ({ page }) => {
            await logisticRegressionPage.setNormalization('Z-Score');
            await logisticRegressionPage.addTransformation('Polynomial', { degree: 3 });

            await expect(page.getByText('Output features: 7')).toBeVisible();

            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            const transformationExpected = EXPECTED_RESULTS.withTransformation;
            await logisticRegressionPage.verifyTrainingResults(
                EXPECTED_RESULTS.withTransformation.trainAccuracy,
                EXPECTED_RESULTS.withTransformation.testAccuracy,
            );
            await logisticRegressionPage.verifyConfusionMatrixMetrics(
                transformationExpected.confusionMatrixMetrics,
            );
            await logisticRegressionPage.verifyROCAUC(transformationExpected.rocAuc);
            await logisticRegressionPage.verifyLearnedParameters(
                transformationExpected.learnedParameters,
            );
        });
    });

    test.describe('Multiclass Classification (Softmax)', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 70.83%',
                testAccuracy: 'Test Accuracy: 41.67%',
                confusionMatrixMetrics: {
                    Accuracy: '70.8%',
                    MCC: '0.614',
                    "Cohen's Kappa": '0.540',
                    'Macro Precision': '83.9%',
                    'Macro Recall': '65.3%',
                    'Macro F1': '59.6%',
                    'Weighted Precision': '70.8%',
                    'Weighted Recall': '70.8%',
                    'Weighted F1': '63.6%',
                },
                multiclassROCAUC: { macroAuc: '0.999', weightedAuc: '0.999' },
                learnedParameters: [
                    { bias: '0.0353', weights: { x1: '+0.3280', x2: '+0.4059' } },
                    { bias: '0.1551', weights: { x1: '+0.1198', x2: '+0.0734' } },
                    { bias: '-0.1903', weights: { x1: '+0.8159', x2: '+0.0166' } },
                ],
            },
            withMomentum: {
                trainAccuracy: 'Train Accuracy: 97.92%',
                testAccuracy: 'Test Accuracy: 100.00%',
                confusionMatrixMetrics: {
                    Accuracy: '97.9%',
                    MCC: '0.969',
                    "Cohen's Kappa": '0.968',
                    'Macro Precision': '98.2%',
                    'Macro Recall': '98.0%',
                    'Macro F1': '98.1%',
                    'Weighted Precision': '97.9%',
                    'Weighted Recall': '97.9%',
                    'Weighted F1': '97.9%',
                },
                multiclassROCAUC: { macroAuc: '1.000', weightedAuc: '1.000' },
                learnedParameters: [
                    { bias: '-1.4710', weights: { x1: '-0.4209', x2: '+0.8417' } },
                    { bias: '2.7337', weights: { x1: '-0.4191', x2: '-0.6303' } },
                    { bias: '-1.2628', weights: { x1: '+0.7662', x2: '-0.2403' } },
                ],
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            logisticRegressionPage = new LogisticRegressionPage(page, DATASET_FILE);
            await logisticRegressionPage.navigateToPage();
            await logisticRegressionPage.configureDataset();
            await logisticRegressionPage.setBasicConfiguration();

            // Softmax will be selected by default as DATASET has 3 classes
        });

        test('should be disabled binary classification and enabled softmax', async ({ page }) => {
            await expect(page.getByTestId('classification-type-binary')).toBeDisabled();
            await expect(page.getByTestId('classification-type-softmax')).toBeChecked();
        });

        test('should successfully train model by default', async () => {
            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await logisticRegressionPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await logisticRegressionPage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
            await logisticRegressionPage.verifyMulticlassROCAUC(defaultExpected.multiclassROCAUC);
            await logisticRegressionPage.verifyMulticlassLearnedParameters(
                defaultExpected.learnedParameters,
            );
        });

        test('should successfully train model using Momentum', async () => {
            await logisticRegressionPage.setOptimizer('Momentum');
            await logisticRegressionPage.setLearningRate(0.1);
            await logisticRegressionPage.setMaxIterations(50);
            await logisticRegressionPage.setRegularization('L2 (Ridge)', { lambda: 0.1 });

            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            const momentumExpected = EXPECTED_RESULTS.withMomentum;
            await logisticRegressionPage.verifyTrainingResults(
                momentumExpected.trainAccuracy,
                momentumExpected.testAccuracy,
            );
            await logisticRegressionPage.verifyConfusionMatrixMetrics(
                momentumExpected.confusionMatrixMetrics,
            );
            await logisticRegressionPage.verifyMulticlassROCAUC(momentumExpected.multiclassROCAUC);
            await logisticRegressionPage.verifyMulticlassLearnedParameters(
                momentumExpected.learnedParameters,
            );
        });
    });

    test.describe('Multiclass Classification (One-vs-Rest)', () => {
        const DATASET_FILE = path.resolve(
            __dirname,
            './data/multiclass_logistic_regression_test_data.csv',
        );
        const EXPECTED_RESULTS = {
            default: {
                trainAccuracy: 'Train Accuracy: 72.92%',
                testAccuracy: 'Test Accuracy: 41.67%',
                confusionMatrixMetrics: {
                    Accuracy: '72.9%',
                    MCC: '0.627',
                    "Cohen's Kappa": '0.574',
                    'Macro Precision': '81.8%',
                    'Macro Recall': '67.3%',
                    'Macro F1': '60.1%',
                    'Weighted Precision': '72.9%',
                    'Weighted Recall': '72.9%',
                    'Weighted F1': '64.4%',
                },
                multiclassROCAUC: { macroAuc: '0.987', weightedAuc: '0.986' },
                learnedParameters: [
                    { bias: '-0.1058', weights: { x1: '-0.3955', x2: '+0.3313' } },
                    { bias: '0.0634', weights: { x1: '-0.2890', x2: '-0.3251' } },
                    { bias: '-0.1279', weights: { x1: '+0.4087', x2: '-0.3512' } },
                ],
            },
        } as const;

        test.beforeEach(async ({ page }) => {
            logisticRegressionPage = new LogisticRegressionPage(page, DATASET_FILE);
            await logisticRegressionPage.navigateToPage();
            await logisticRegressionPage.configureDataset();
            await logisticRegressionPage.setBasicConfiguration();

            await logisticRegressionPage.setClassificationType('One-vs-Rest');
        });

        test('should successfully train model by default', async () => {
            await logisticRegressionPage.startTraining();
            await logisticRegressionPage.waitForTrainingCompletion();

            const defaultExpected = EXPECTED_RESULTS.default;
            await logisticRegressionPage.verifyTrainingResults(
                defaultExpected.trainAccuracy,
                defaultExpected.testAccuracy,
            );
            await logisticRegressionPage.verifyConfusionMatrixMetrics(
                defaultExpected.confusionMatrixMetrics,
            );
            await logisticRegressionPage.verifyMulticlassROCAUC(defaultExpected.multiclassROCAUC);
            await logisticRegressionPage.verifyMulticlassLearnedParameters(
                defaultExpected.learnedParameters,
            );
        });
    });
});
