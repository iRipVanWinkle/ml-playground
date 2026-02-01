import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, NaiveBayesParams, NaiveBayesCallbackParameters } from '@/ml/types';
import type { NaiveBayesTrainingReport } from '../types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import {
    type DatasetManager,
    type LiveMetrics,
    getSafeMatrixFromTensor,
    getSafeTensorValue,
    getSafeTensorArray,
    createTensorContainer,
    type TensorContainer,
} from '@/app/shared/workers';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

type MetricsTensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
    loss: Scalar;
};

export class NaiveBayesLiveMetrics
    implements LiveMetrics<NaiveBayesCallbackParameters, NaiveBayesTrainingReport>
{
    private model: Model<NaiveBayesParams>;
    private datasetManager: DatasetManager;

    static factory(model: Model<NaiveBayesParams>, datasetManager: DatasetManager) {
        return new NaiveBayesLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<NaiveBayesParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(
        params: NaiveBayesCallbackParameters,
    ): Promise<NaiveBayesTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();
        const numClasses = this.datasetManager.getNumClasses();

        const { iteration, params: modelParams } = params;

        let yPredictions: Tensor2D | undefined;
        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelParams);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, modelParams);

        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, modelParams)
            : createTensorContainer<MetricsTensors>();

        const [
            predictionPredictedLabels,

            //train
            trainPredictedLabels,
            trainAccuracyValue,
            trainConfusionMatrixValue,
            trainProbabilityValue,
            trainLabelValue,
            //test
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
            type: 'naive-bayes',
            taskType: 'classification',
            testAccuracy: testAccuracyValue ?? 0,
            trainAccuracy: trainAccuracyValue ?? 0,
            trainPredictedLabels: trainPredictedLabels,
            testPredictedLabels: testPredictedLabels,
            predictionPredictedLabels: predictionPredictedLabels,
            iteration: iteration + 1,
            params: modelParams,

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

    private evaluateMetrics(
        X: Tensor2D,
        yTrue: Tensor2D,
        params: NaiveBayesParams,
    ): TensorContainer<MetricsTensors> {
        const numClasses = this.datasetManager.getNumClasses();

        const trainPredictWithProbs = this.model.predictWithMetadata(X, params);
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
