import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { KNNCallbackParameters, KNNParams, Model } from '@/ml/types';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorArray,
    getSafeTensorValue,
    type DatasetManager,
    type LiveMetrics,
    type TensorContainer,
} from '@/app/shared/workers';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';
import type { KNNClassificationTrainingReport } from '../types';

type MetricsTensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
};

export class KNNClassificationLiveMetrics
    implements LiveMetrics<KNNCallbackParameters, KNNClassificationTrainingReport>
{
    private model: Model<KNNParams>;
    private datasetManager: DatasetManager;

    static factory(model: Model<KNNParams>, datasetManager: DatasetManager) {
        return new KNNClassificationLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<KNNParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(
        params: KNNCallbackParameters,
    ): Promise<KNNClassificationTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();
        const numClasses = this.datasetManager.getNumClasses();

        const { params: modelParams } = params;

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

        yPredictions?.dispose();
        train.dispose();
        test.dispose();

        return {
            type: 'knn',
            taskType: 'classification',
            testAccuracy: testAccuracyValue ?? 0,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels,
            predictionPredictedLabels: predictionPredictedLabels,
            params: modelParams,
            trainConfusionMatrix: confusionMatrixData(trainConfusionMatrixValue!, numClasses),
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
        modelParams: KNNParams,
    ): TensorContainer<MetricsTensors> {
        const numClasses = this.datasetManager.getNumClasses();

        const result = this.model.predictWithMetadata(X, modelParams);

        if (result.type !== 'classification') {
            throw new Error('Model is not a classification model');
        }

        const metrics = createTensorContainer<MetricsTensors>();
        metrics.y = result.predictions;
        metrics.probabilities = result.probabilities;
        metrics.accuracy = accuracy(yTrue, metrics.y);
        metrics.confusionMatrix = confusionMatrix(yTrue, metrics.y, numClasses);

        return metrics;
    }
}
