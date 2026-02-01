import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import type { DatasetManager, LiveMetrics, TensorContainer } from '@/app/shared/workers';
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

type MetricsTensors = {
    y: Tensor2D;
    probabilities: Tensor2D;
    accuracy: Scalar;
    confusionMatrix: Tensor2D;
    loss: Scalar;
};

export class TreeClassificationLiveMetrics
    implements LiveMetrics<TreeCallbackParameters, TreeClassificationTrainingReport>
{
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

    async calculateMetrics(
        params: TreeCallbackParameters,
    ): Promise<TreeClassificationTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const { threadId, tree } = params;

        this.trees[threadId] = tree;

        let yPredictions: Tensor2D | undefined;
        if (predictionData) {
            yPredictions = this.model.predict(predictionData, this.trees);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, this.trees);
        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, this.trees)
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

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        train.dispose();
        test.dispose();

        return {
            type: 'tree',
            taskType: 'classification',
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            params: this.trees,
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

    private evaluateMetrics(
        X: Tensor2D,
        yTrue: Tensor2D,
        trees: EnsembleTree,
    ): TensorContainer<MetricsTensors> {
        const numClasses = this.datasetManager.getNumClasses();

        const trainPredictWithProbs = this.model.predictWithMetadata(X, trees);
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
