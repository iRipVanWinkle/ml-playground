import { Rank, variable, Variable, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, ModelRepresentation, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { LinearTrainingReport } from '../types';
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

export class LinearLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, LinearTrainingReport>
{
    private lossHistory: number[] = [];
    private iterationCount: number = 0;
    private theta?: Variable<Rank.R2>;

    private model: Model<ModelRepresentation>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new LinearLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: OptimizerCallbackParameters): void {
        const { iteration, theta, loss } = params;

        this.lossHistory.push(loss);
        this.iterationCount = iteration + 1;

        if (!this.theta) {
            this.theta = variable(theta);
        }
        this.theta.assign(theta);
    }

    async calculateMetrics(): Promise<LinearTrainingReport> {
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

        train.y = this.model.predict(trainingData.X, theta);
        train.mae = meanAbsoluteError(trainingData.y, train.y);
        train.mse = meanSquaredError(trainingData.y, train.y);
        train.rmse = rootMeanSquaredError(trainingData.y, train.y);
        train.r2 = r2Score(trainingData.y, train.y);
        train.residuals = residuals(trainingData.y, train.y);

        if (testData) {
            const [yTesting, , testLoss] = this.model.evaluate(testData.X, testData.y, theta);
            test.y = yTesting;
            test.loss = testLoss;
            test.mae = meanAbsoluteError(testData.y, test.y);
            test.mse = meanSquaredError(testData.y, test.y);
            test.rmse = rootMeanSquaredError(testData.y, test.y);
            test.r2 = r2Score(testData.y, test.y);
            test.residuals = residuals(testData.y, test.y);
        }

        // Transpose theta for speedup rendering on UI side
        const transposedTheta = theta.transpose() as Tensor2D;

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
            getMatrixFromTensor(transposedTheta),
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
        transposedTheta.dispose();
        train.dispose();
        test.dispose();

        const hasTestMetrics =
            testMaeValue !== undefined &&
            testMseValue !== undefined &&
            testRmseValue !== undefined &&
            testR2Value !== undefined;

        return {
            type: 'linear',
            taskType: 'regression',
            trainLossHistory: [this.lossHistory],
            iteration: this.iterationCount,
            trainLoss: this.lossHistory.at(-1) ?? 0,
            testLoss: testLossValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels,
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
