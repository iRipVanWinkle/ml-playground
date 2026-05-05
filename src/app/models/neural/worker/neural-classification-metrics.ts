import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics, TensorContainer } from '@/app/shared/workers';
import type { NeuralClassificationTrainingReport } from '../types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorArray,
    getSafeTensorValue,
} from '@/app/shared/workers';
import { createEmptyMatrix } from '@/app/shared/helpers';

type MetricsTensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
    loss: Scalar;
};

export class NeuralClassificationLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, NeuralClassificationTrainingReport>
{
    private lossHistory: number[] = [];
    private theta?: Tensor2D;

    private model: Model<Tensor2D>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new NeuralClassificationLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(
        params: OptimizerCallbackParameters,
    ): Promise<NeuralClassificationTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const { iteration, theta, loss } = params;

        this.lossHistory.push(loss);
        this.theta = theta;

        let yPredictions: Tensor2D | undefined;
        if (predictionData) {
            yPredictions = this.model.predict(predictionData, theta);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, theta);
        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, theta)
            : createTensorContainer<MetricsTensors, 'partial'>();

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
            type: 'neural',
            taskType: 'classification',
            trainLossHistory: [this.lossHistory],
            iteration: iteration + 1,
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            theta: createEmptyMatrix(), // For now it isn't handled on UI side, so we can return an empty matrix
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

    dispose(): void {
        this.theta?.dispose();
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
