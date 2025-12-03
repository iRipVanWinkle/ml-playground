import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { TreeRegressionTrainingReport } from '../types';
import { getMatrixFromTensor } from '@/ml/matrix';
import { getSafeMatrixFromTensor, getSafeTensorValue } from '@/app/shared/workers';

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

    getIterations(): number[] {
        return [...this.iterationCounts];
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

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );
        }

        const [
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            // test
            testPredictedLabels,
            testLossValue,
        ] = await Promise.all([
            getSafeMatrixFromTensor(yPredictions),
            // train
            getMatrixFromTensor(yTraining),
            // test
            getSafeMatrixFromTensor(yTesting),
            getSafeTensorValue(testLoss),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        yTraining?.dispose();
        yTestingProbability?.dispose();
        yTesting?.dispose();
        yTrainingProbability?.dispose();
        trainLoss?.dispose();
        testLoss?.dispose();
        // Tree models don't need disposal like tensor models

        return {
            type: 'tree',
            taskType: 'regression',
            iterations: this.getIterations(),
            testLoss: testLossValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
        };
    }

    dispose(): void {
        this.thetaArray.forEach((theta) => theta.dispose());
    }
}
