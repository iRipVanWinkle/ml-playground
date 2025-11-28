import { type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, NaiveBayesParams, NaiveBayesCallbackParameters } from '@/ml/types';
import {
    getSafeMatrixFromTensor,
    type DatasetManager,
    type LiveMetrics,
    getSafeTensorValue,
    getSafeTensorArray,
} from '@/app/shared/workers';
import type { NaiveBayesTrainingReport } from '../types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

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

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;
        let testAccuracy: Scalar | undefined;
        let testConfusionMatrix: Tensor2D | undefined;

        // Model uses its internal params (no need to pass them)
        if (predictionData) {
            yPredictions = this.model.predict(predictionData, this.params);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            this.params,
        );
        const trainAccuracyMetric = accuracy(trainingData.y, yTraining!);
        const trainConfusionMatrix = confusionMatrix(trainingData.y, yTraining, numClasses);

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                this.params,
            );
            testAccuracy = accuracy(testData.y, yTesting!);
            testConfusionMatrix = confusionMatrix(testData.y, yTesting!, numClasses);
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
            getSafeMatrixFromTensor(yTraining),
            getSafeTensorValue(trainAccuracyMetric),
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
        trainAccuracyMetric?.dispose();

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
