import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { TreeClassificationTrainingReport } from '../types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorArray,
    getSafeTensorValue,
} from '@/app/shared/workers';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

type Tensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
    loss: Scalar;
};

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
        const numClasses = this.datasetManager.getNumClasses();

        const train = createTensorContainer<Tensors>();
        const test = createTensorContainer<Tensors, 'partial'>();

        let yPredictions: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );
        train.y = yTraining;
        train.probabilities = yTrainingProbability;
        train.loss = trainLoss;
        train.accuracy = accuracy(trainingData.y, yTraining);
        train.confusionMatrix = confusionMatrix(trainingData.y, yTraining, numClasses);

        if (testData) {
            const [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );

            test.y = yTesting;
            test.probabilities = yTestingProbability;
            test.loss = testLoss;
            test.accuracy = accuracy(testData.y, yTesting);
            test.confusionMatrix = confusionMatrix(testData.y, yTesting, numClasses);
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
            getSafeMatrixFromTensor(train.y),
            getSafeTensorValue(train.accuracy),
            getSafeTensorArray(train.confusionMatrix),
            getSafeMatrixFromTensor(train.probabilities),
            getSafeMatrixFromTensor(trainingData.y),
            // test
            getSafeMatrixFromTensor(test.y),
            getSafeTensorValue(test.accuracy),
            getSafeTensorArray(test.confusionMatrix),
            getSafeMatrixFromTensor(test.probabilities),
            getSafeMatrixFromTensor(testData?.y),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        train.dispose();
        test.dispose();

        return {
            type: 'tree',
            taskType: 'classification',
            iterations: this.getIterations(),
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            params: modelRepresentation,
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
