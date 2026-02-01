import { concat, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics, TensorContainer } from '@/app/shared/workers';
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

type MetricsTensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
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

    async calculateMetrics(params: OptimizerCallbackParameters): Promise<LogisticTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();
        const numClasses = this.datasetManager.getNumClasses();

        const { threadId, iteration, theta, loss } = params;

        this.updateIteration(threadId, iteration);
        this.updateLossHistory(threadId, loss);
        const combinedTheta = this.storeAndMergeThreadTheta(threadId, theta);

        let yPredictions: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, combinedTheta);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, combinedTheta);
        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, combinedTheta)
            : createTensorContainer<MetricsTensors, 'partial'>();

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
            getSafeMatrixFromTensor(combinedTheta),
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
        combinedTheta.dispose();
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

    private updateIteration(threadId: number, iteration: number): void {
        for (let i = 0; i <= threadId; i++) {
            this.iterationCounts[i] = this.iterationCounts[i] ?? 0;
        }
        this.iterationCounts[threadId] = iteration + 1;
    }

    private updateLossHistory(threadId: number, loss: number): void {
        for (let i = 0; i <= threadId; i++) {
            this.lossHistory[i] = this.lossHistory[i] ?? [];
        }
        this.lossHistory[threadId].push(loss);
    }

    private storeAndMergeThreadTheta(threadId: number, theta: Tensor2D): Tensor2D {
        this.thetaArray[threadId] = theta;
        return concat(this.thetaArray.filter(Boolean), 1) as Tensor2D;
    }

    private evaluateMetrics(
        X: Tensor2D,
        yTrue: Tensor2D,
        theta: Tensor2D,
    ): TensorContainer<MetricsTensors> {
        const numClasses = this.datasetManager.getNumClasses();

        const trainPredictWithProbs = this.model.predictWithMetadata(X, theta);
        if (trainPredictWithProbs.type !== 'classification') {
            throw new Error('Model is not a classification model');
        }

        const metrics = createTensorContainer<MetricsTensors>();
        metrics.y = trainPredictWithProbs.predictions;
        metrics.probabilities = trainPredictWithProbs.probabilities;
        metrics.accuracy = accuracy(yTrue, metrics.y);
        metrics.confusionMatrix = confusionMatrix(yTrue, metrics.y, numClasses);

        return metrics;
    }
}
