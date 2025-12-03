import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { TreeClassificationTrainingReport } from '../types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import {
    getSafeMatrixFromTensor,
    getSafeTensorArray,
    getSafeTensorValue,
} from '@/app/shared/workers';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

export class TreeClassificationLiveMetrics
    implements LiveMetrics<TreeCallbackParameters, TreeClassificationTrainingReport>
{
    private iterationCounts: number[] = [];
    private model: Model<EnsembleTree>;
    private datasetManager: DatasetManager;

    private trees: TreeNode[] = [];

    static factory(model: Model<EnsembleTree>, datasetManager: DatasetManager) {
        return new TreeClassificationLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<EnsembleTree>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: TreeCallbackParameters): void {
        const { threadId, iteration, tree } = params;

        this.trees[threadId] = tree;

        this.iterationCounts[threadId] = this.iterationCounts[threadId] ?? 0;
        this.iterationCounts[threadId] = iteration + 1;
    }

    getIterations(): number[] {
        return [...this.iterationCounts];
    }

    getModelRepresentation(): EnsembleTree {
        return this.trees;
    }

    async calculateMetrics(): Promise<TreeClassificationTrainingReport> {
        const modelRepresentation = this.getModelRepresentation();

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;
        let testAccuracy: Scalar | undefined;
        let testConfusionMatrix: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );

        const trainAccuracy = accuracy(trainingData.y, yTraining!);
        const trainConfusionMatrix = confusionMatrix(
            trainingData.y,
            yTraining,
            this.datasetManager.getNumClasses(),
        );

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );

            testAccuracy = accuracy(testData.y, yTesting!);
            testConfusionMatrix = confusionMatrix(
                testData.y,
                yTesting!,
                this.datasetManager.getNumClasses(),
            );
        }

        const [
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            trainAccuracyValue,
            trainConfusionMatrixValue,
            trainProbabilityValue,
            trainLabelValue,
            // test
            testPredictedLabels,
            testAccuracyValue,
            testConfusionMatrixValue,
            testProbabilityValue,
            testLabelValue,
        ] = await Promise.all([
            getSafeMatrixFromTensor(yPredictions),
            // train
            getSafeMatrixFromTensor(yTraining),
            getSafeTensorValue(trainAccuracy),
            getSafeTensorArray(trainConfusionMatrix),
            getSafeMatrixFromTensor(yTrainingProbability),
            getSafeMatrixFromTensor(trainingData.y),
            // test
            getSafeMatrixFromTensor(yTesting),
            getSafeTensorValue(testAccuracy),
            getSafeTensorArray(testConfusionMatrix),
            getSafeMatrixFromTensor(yTestingProbability),
            getSafeMatrixFromTensor(testData?.y),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        yTraining?.dispose();
        yTestingProbability?.dispose();
        yTesting?.dispose();
        yTrainingProbability?.dispose();
        trainLoss?.dispose();
        testLoss?.dispose();
        // Tree models don't need disposal like tensor models

        return {
            type: 'tree',
            taskType: 'classification',
            iterations: this.getIterations(),
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            trainConfusionMatrix: confusionMatrixData(
                trainConfusionMatrixValue!,
                this.datasetManager.getNumClasses(),
            ),
            testConfusionMatrix: testConfusionMatrixValue
                ? confusionMatrixData(testConfusionMatrixValue, this.datasetManager.getNumClasses())
                : undefined,

            trainRocCurve: rocCurveData(
                trainLabelValue,
                trainProbabilityValue,
                trainConfusionMatrixValue,
            ),
            testRocCurve:
                testLabelValue && testProbabilityValue && testConfusionMatrixValue
                    ? rocCurveData(testLabelValue, testProbabilityValue, testConfusionMatrixValue)
                    : undefined,
        };
    }
}
