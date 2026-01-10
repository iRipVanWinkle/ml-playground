import { concat, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { LogisticTrainingReport } from '../types';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorArray,
    getSafeTensorValue,
} from '@/app/shared/workers';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

function fixLength(matrix: number[][]): number[][] {
    const minLength = Math.min(...matrix.map((m) => m.length)) ?? 0;
    return matrix.map((m) => m.slice(0, minLength));
}

type Tensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
    loss: Scalar;
};

export class LogisticLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, LogisticTrainingReport>
{
    private lossHistory: number[][] = [];
    private iterationCounts: number[] = [];
    private thetaArray: Tensor2D[] = [];

    private model: Model<Tensor2D>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new LogisticLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: OptimizerCallbackParameters): void {
        const { threadId, iteration, theta, loss } = params;

        this.thetaArray[threadId] = theta;

        for (let i = 0; i <= threadId; i++) {
            this.lossHistory[i] = this.lossHistory[i] ?? [];
        }
        this.lossHistory[threadId].push(loss);

        for (let i = 0; i <= threadId; i++) {
            this.iterationCounts[i] = this.iterationCounts[i] ?? 0;
        }
        this.iterationCounts[threadId] = iteration + 1;
    }

    async calculateMetrics(): Promise<LogisticTrainingReport> {
        const modelRepresentation = concat(this.thetaArray.filter(Boolean), 1) as Tensor2D;

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
            thetaArray,
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
            getSafeMatrixFromTensor(modelRepresentation),
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
        modelRepresentation.dispose();
        train.dispose();
        test.dispose();

        return {
            type: 'logistic',
            taskType: 'classification',
            trainLossHistory: fixLength(this.lossHistory),
            iterations: this.iterationCounts,
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            theta: thetaArray!,
            trainConfusionMatrix: confusionMatrixData(trainConfusionMatrixValue, numClasses),
            testConfusionMatrix: testConfusionMatrixValue
                ? confusionMatrixData(testConfusionMatrixValue, numClasses)
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

    dispose(): void {
        this.thetaArray.forEach((theta) => theta.dispose());
    }
}
