import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { TreeRegressionTrainingReport } from '../types';
import { getMatrixFromTensor } from '@/ml/matrix';
import { getSafeMatrixFromTensor, getSafeTensorValue } from '@/app/shared/workers';
import {
    meanAbsoluteError,
    meanSquaredError,
    r2Score,
    residuals,
    rootMeanSquaredError,
} from '@/ml/metrics';

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

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;
        let testMae: Scalar | undefined;
        let testMse: Scalar | undefined;
        let testRmse: Scalar | undefined;
        let testR2: Scalar | undefined;
        let testResiduals: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );
        trainLoss.print();
        const trainMae = meanAbsoluteError(trainingData.y, yTraining!);
        const trainMse = meanSquaredError(trainingData.y, yTraining!);
        const trainRmse = rootMeanSquaredError(trainingData.y, yTraining!);
        const trainR2 = r2Score(trainingData.y, yTraining!);
        const trainResiduals = residuals(trainingData.y, yTraining!);

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );
            testMae = meanAbsoluteError(testData.y, yTesting!);
            testMse = meanSquaredError(testData.y, yTesting!);
            testRmse = rootMeanSquaredError(testData.y, yTesting!);
            testR2 = r2Score(testData.y, yTesting!);
            testResiduals = residuals(testData.y, yTesting!);
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
            getMatrixFromTensor(yTraining),
            getSafeTensorValue(trainMae),
            getSafeTensorValue(trainMse),
            getSafeTensorValue(trainRmse),
            getSafeTensorValue(trainR2),
            getSafeMatrixFromTensor(trainResiduals),
            // test
            getSafeMatrixFromTensor(yTesting),
            getSafeTensorValue(testLoss),
            getSafeTensorValue(testMae),
            getSafeTensorValue(testMse),
            getSafeTensorValue(testRmse),
            getSafeTensorValue(testR2),
            getSafeMatrixFromTensor(testResiduals),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        yTraining?.dispose();
        yTestingProbability?.dispose();
        yTesting?.dispose();
        yTrainingProbability?.dispose();
        trainLoss?.dispose();
        testLoss?.dispose();
        trainResiduals?.dispose();
        trainMae?.dispose();
        trainMse?.dispose();
        trainRmse?.dispose();
        trainR2?.dispose();
        testResiduals?.dispose();
        testMae?.dispose();
        testMse?.dispose();
        testRmse?.dispose();
        testR2?.dispose();

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
