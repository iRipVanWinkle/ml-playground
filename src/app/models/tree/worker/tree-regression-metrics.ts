import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorValue,
    type DatasetManager,
    type LiveMetrics,
} from '@/app/shared/workers';
import { getMatrixFromTensor } from '@/ml/matrix';
import {
    meanAbsoluteError,
    meanSquaredError,
    r2Score,
    residuals,
    rootMeanSquaredError,
} from '@/ml/metrics';
import type { TreeRegressionTrainingReport } from '../types';

type Tensors = {
    y: Tensor2D;
    loss: Scalar;
    mae: Scalar;
    mse: Scalar;
    rmse: Scalar;
    r2: Scalar;
    residuals: Tensor2D;
};

export class TreeRegressionLiveMetrics
    implements LiveMetrics<TreeCallbackParameters, TreeRegressionTrainingReport>
{
    private iterationCounts: number[] = [];
    private thetaArray: Tensor2D[] = [];

    private trees: TreeNode[] = [];

    private model: Model<EnsembleTree>;
    private datasetManager: DatasetManager;

    static factory(model: Model<EnsembleTree>, datasetManager: DatasetManager) {
        return new TreeRegressionLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<EnsembleTree>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: TreeCallbackParameters): void {
        const { threadId, iteration, tree } = params;

        this.trees[threadId] = tree;

        this.iterationCounts[threadId] = this.iterationCounts[threadId] ?? 0;
        this.iterationCounts[threadId] = iteration + 1;
    }

    getModelRepresentation(): EnsembleTree {
        return this.trees;
    }

    async calculateMetrics(): Promise<TreeRegressionTrainingReport> {
        const modelRepresentation = this.getModelRepresentation();

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const train = createTensorContainer<Tensors>();
        const test = createTensorContainer<Tensors, 'partial'>();

        let yPredictions: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, , trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );
        train.y = yTraining;
        train.loss = trainLoss;
        train.mae = meanAbsoluteError(trainingData.y, yTraining);
        train.mse = meanSquaredError(trainingData.y, yTraining);
        train.rmse = rootMeanSquaredError(trainingData.y, yTraining);
        train.r2 = r2Score(trainingData.y, yTraining);
        train.residuals = residuals(trainingData.y, yTraining);

        if (testData) {
            const [yTesting, , testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );
            test.y = yTesting;
            test.loss = testLoss;
            test.mae = meanAbsoluteError(testData.y, yTesting);
            test.mse = meanSquaredError(testData.y, yTesting);
            test.rmse = rootMeanSquaredError(testData.y, yTesting);
            test.r2 = r2Score(testData.y, yTesting);
            test.residuals = residuals(testData.y, yTesting);
        }

        const [
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
            type: 'tree',
            taskType: 'regression',
            iterations: this.iterationCounts,
            testLoss: testLossValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            params: modelRepresentation,
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
        this.thetaArray.forEach((theta) => theta.dispose());
    }
}
