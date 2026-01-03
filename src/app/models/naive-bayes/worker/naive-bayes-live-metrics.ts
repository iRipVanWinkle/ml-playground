import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, NaiveBayesParams, NaiveBayesCallbackParameters } from '@/ml/types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import {
    type DatasetManager,
    type LiveMetrics,
    getSafeMatrixFromTensor,
    getSafeTensorValue,
    getSafeTensorArray,
    createTensorContainer,
} from '@/app/shared/workers';
import type { NaiveBayesTrainingReport } from '../types';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

type Tensors = {
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
    private iteration = 0;
    private params?: NaiveBayesParams;

    static factory(model: Model<NaiveBayesParams>, datasetManager: DatasetManager) {
        return new NaiveBayesLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<NaiveBayesParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: NaiveBayesCallbackParameters): void {
        this.iteration = params.iteration;
        this.params = params.params;
    }

    async calculateMetrics(): Promise<NaiveBayesTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();
        const numClasses = this.datasetManager.getNumClasses();

        const train = createTensorContainer<Tensors>();
        const test = createTensorContainer<Tensors, 'partial'>();

        let yPredictions: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, this.params);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            this.params,
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
                this.params,
            );
            test.y = yTesting;
            test.probabilities = yTestingProbability;
            test.loss = testLoss;
            test.accuracy = accuracy(testData.y, yTesting);
            test.confusionMatrix = confusionMatrix(testData.y, yTesting, numClasses);
        }

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
            iteration: this.iteration,
            params: this.params!,

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
        // No tensors to dispose
    }
}
