import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { NeuralRegressionTrainingReport } from '../types';
import { getMatrixFromTensor } from '@/ml/matrix';
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

type Tensors = {
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
    private lossHistory: number[] = [];
    private iterationCount = 0;
    private theta?: Tensor2D;

    private model: Model<Tensor2D>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new NeuralRegressionLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: OptimizerCallbackParameters): void {
        const { iteration, theta, loss } = params;

        this.lossHistory.push(loss);
        this.iterationCount = iteration + 1;
        this.theta = theta;
    }

    async calculateMetrics(): Promise<NeuralRegressionTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const theta = this.theta!;

        const train = createTensorContainer<Tensors>();
        const test = createTensorContainer<Tensors, 'partial'>();

        let yPredictions: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, theta);
        }

        const [yTraining, , trainLoss] = this.model.evaluate(trainingData.X, trainingData.y, theta);
        train.y = yTraining;
        train.loss = trainLoss;
        train.mae = meanAbsoluteError(trainingData.y, yTraining);
        train.mse = meanSquaredError(trainingData.y, yTraining);
        train.rmse = rootMeanSquaredError(trainingData.y, yTraining);
        train.r2 = r2Score(trainingData.y, yTraining);
        train.residuals = residuals(trainingData.y, yTraining);

        if (testData) {
            const [yTesting, , testLoss] = this.model.evaluate(testData.X, testData.y, theta);
            test.y = yTesting;
            test.loss = testLoss;
            test.mae = meanAbsoluteError(testData.y, yTesting);
            test.mse = meanSquaredError(testData.y, yTesting);
            test.rmse = rootMeanSquaredError(testData.y, yTesting);
            test.r2 = r2Score(testData.y, yTesting);
            test.residuals = residuals(testData.y, yTesting);
        }

        const [
            thetaArray,
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
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
            getMatrixFromTensor(theta),
            getSafeMatrixFromTensor(yPredictions),
            // train
            getMatrixFromTensor(train.y),
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
            iteration: this.iterationCount,
            trainLoss: this.lossHistory?.at(-1) ?? 0,
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

    dispose(): void {
        this.theta?.dispose();
    }
}
