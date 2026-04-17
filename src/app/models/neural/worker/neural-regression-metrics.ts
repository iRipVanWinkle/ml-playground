import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics, TensorContainer } from '@/app/shared/workers';
import type { NeuralRegressionTrainingReport, NeuralSettings } from '../types';
import type { TrainingSettings } from '../../types';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorValue,
} from '@/app/shared/workers';
import {
    meanAbsoluteError,
    meanSquaredError,
    r2Score,
    residuals,
    rootMeanSquaredError,
} from '@/ml/metrics';
import { lossFunctionFactory } from '@/ml/factories';

type MetricsTensors = {
    y: Tensor2D;
    loss: Scalar;
    mae: Scalar;
    mse: Scalar;
    rmse: Scalar;
    r2: Scalar;
    residuals: Tensor2D;
};

export class NeuralRegressionLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, NeuralRegressionTrainingReport>
{
    private model: Model<Tensor2D>;
    private datasetManager: DatasetManager;
    private modelSettings: NeuralSettings;

    private lossHistory: number[] = [];

    static factory(
        model: Model<Tensor2D>,
        datasetManager: DatasetManager,
        settings: TrainingSettings,
    ) {
        const { modelSettings } = settings;

        if (modelSettings.type !== 'neural') {
            throw new Error(
                `Invalid settings type: expected 'neural', got '${modelSettings.type}'`,
            );
        }

        return new NeuralRegressionLiveMetrics(modelSettings, model, datasetManager);
    }

    private constructor(
        modelSettings: NeuralSettings,
        model: Model<Tensor2D>,
        datasetManager: DatasetManager,
    ) {
        this.model = model;
        this.datasetManager = datasetManager;
        this.modelSettings = modelSettings;
    }

    async calculateMetrics(
        params: OptimizerCallbackParameters,
    ): Promise<NeuralRegressionTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const { iteration, theta, loss } = params;
        this.lossHistory.push(loss);

        let yPredictions: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, theta);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, theta);
        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, theta)
            : createTensorContainer<MetricsTensors, 'partial'>();

        const [
            thetaArray,
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
            getSafeMatrixFromTensor(theta),
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

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        train.dispose();
        test.dispose();

        const hasTestMetrics =
            testMaeValue !== undefined &&
            testMseValue !== undefined &&
            testRmseValue !== undefined &&
            testR2Value !== undefined;

        return {
            type: 'neural',
            taskType: 'regression',
            trainLossHistory: [this.lossHistory],
            iteration: iteration + 1,
            optimizerLoss: this.lossHistory?.at(-1) ?? 0,
            trainLoss: trainLossValue,
            testLoss: testLossValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            theta: thetaArray,
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
        theta: Tensor2D,
    ): TensorContainer<MetricsTensors> {
        const metrics = createTensorContainer<MetricsTensors>();

        metrics.y = this.model.predict(X, theta);
        metrics.mae = meanAbsoluteError(yTrue, metrics.y);
        metrics.mse = meanSquaredError(yTrue, metrics.y);
        metrics.rmse = rootMeanSquaredError(yTrue, metrics.y);
        metrics.r2 = r2Score(yTrue, metrics.y);
        metrics.residuals = residuals(yTrue, metrics.y);

        const lossFunc = lossFunctionFactory(this.modelSettings.lossFunction);
        metrics.loss = lossFunc.compute(yTrue, metrics.y);

        return metrics;
    }
}
