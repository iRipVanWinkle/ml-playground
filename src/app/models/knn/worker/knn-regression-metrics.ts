import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { KNNCallbackParameters, KNNParams, Model } from '@/ml/types';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorValue,
    type DatasetManager,
    type LiveMetrics,
    type TensorContainer,
} from '@/app/shared/workers';
import {
    meanAbsoluteError,
    meanSquaredError,
    r2Score,
    residuals,
    rootMeanSquaredError,
} from '@/ml/metrics';
import type { KNNRegressionTrainingReport } from '../types';
import type { TrainingSettings } from '../../types';

type MetricsTensors = {
    y: Tensor2D;
    loss: Scalar;
    mae: Scalar;
    mse: Scalar;
    rmse: Scalar;
    r2: Scalar;
    residuals: Tensor2D;
};

export class KNNRegressionLiveMetrics
    implements LiveMetrics<KNNCallbackParameters, KNNRegressionTrainingReport>
{
    private model: Model<KNNParams>;
    private datasetManager: DatasetManager;

    static factory(
        model: Model<KNNParams>,
        datasetManager: DatasetManager,
        settings: TrainingSettings,
    ) {
        const { modelSettings } = settings;

        if (modelSettings.type !== 'knn') {
            throw new Error(`Invalid settings type: expected 'knn', got '${modelSettings.type}'`);
        }

        return new KNNRegressionLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<KNNParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(params: KNNCallbackParameters): Promise<KNNRegressionTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const { params: modelParams } = params;

        let yPredictions: Tensor2D | undefined;
        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelParams);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, modelParams);
        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, modelParams)
            : createTensorContainer<MetricsTensors, 'partial'>();

        const [
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            trainLossValue,
            trainMaeValue,
            trainMseValue,
            trainRmseValue,
            trainR2Value,
            trainResidualsArray,
            // test
            testPredictedLabels,
            testLossValue,
            testMaeValue,
            testMseValue,
            testRmseValue,
            testR2Value,
            testResidualsArray,
        ] = await Promise.all([
            getSafeMatrixFromTensor(yPredictions),
            // train
            getSafeMatrixFromTensor(train.y),
            getSafeTensorValue(train.loss),
            getSafeTensorValue(train.mae),
            getSafeTensorValue(train.mse),
            getSafeTensorValue(train.rmse),
            getSafeTensorValue(train.r2),
            getSafeMatrixFromTensor(train.residuals),
            // test
            getSafeMatrixFromTensor(test.y),
            getSafeTensorValue(test.loss),
            getSafeTensorValue(test.mae),
            getSafeTensorValue(test.mse),
            getSafeTensorValue(test.rmse),
            getSafeTensorValue(test.r2),
            getSafeMatrixFromTensor(test.residuals),
        ]);

        yPredictions?.dispose();
        train.dispose();
        test.dispose();

        const hasTestMetrics =
            testMaeValue !== undefined &&
            testMseValue !== undefined &&
            testRmseValue !== undefined &&
            testR2Value !== undefined;

        return {
            type: 'knn',
            taskType: 'regression',
            trainLoss: trainLossValue,
            testLoss: testLossValue,
            trainPredictedLabels: trainPredictedLabels,
            testPredictedLabels: testPredictedLabels,
            predictionPredictedLabels: predictionPredictedLabels,
            params: modelParams,
            trainMetrics: {
                mae: trainMaeValue,
                mse: trainMseValue,
                rmse: trainRmseValue,
                r2: trainR2Value,
            },
            testMetrics: hasTestMetrics
                ? {
                      mae: testMaeValue,
                      mse: testMseValue,
                      rmse: testRmseValue,
                      r2: testR2Value,
                  }
                : undefined,
            trainResiduals: trainResidualsArray,
            testResiduals: testResidualsArray,
        };
    }

    private evaluateMetrics(
        X: Tensor2D,
        yTrue: Tensor2D,
        modelParams: KNNParams,
    ): TensorContainer<MetricsTensors> {
        const metrics = createTensorContainer<MetricsTensors>();

        metrics.y = this.model.predict(X, modelParams);
        metrics.mae = meanAbsoluteError(yTrue, metrics.y);
        metrics.mse = meanSquaredError(yTrue, metrics.y);
        metrics.rmse = rootMeanSquaredError(yTrue, metrics.y);
        metrics.r2 = r2Score(yTrue, metrics.y);
        metrics.residuals = residuals(yTrue, metrics.y);
        metrics.loss = meanSquaredError(yTrue, metrics.y);

        return metrics;
    }
}
