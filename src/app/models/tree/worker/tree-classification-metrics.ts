import { type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { TreeClassificationTrainingReport } from '../types';
import { accuracy } from '@/ml/metrics';

function getTensorArray(
    tensor?: Tensor2D,
    defaultValue?: number[][],
): Promise<number[][] | undefined> {
    if (tensor) {
        return tensor.array();
    }

    return Promise.resolve(defaultValue);
}

async function getTensorData(tensor?: Scalar, defaultValue?: number): Promise<number | undefined> {
    if (tensor) {
        const data = await tensor.data();
        return data[0];
    }

    return Promise.resolve(defaultValue);
}

export class TreeClassificationLiveMetrics
    implements LiveMetrics<TreeCallbackParameters, TreeClassificationTrainingReport>
{
    private iterationCounts: number[] = [];
    private thetaArray: Tensor2D[] = [];

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

    async calculateMetrics(): Promise<TreeClassificationTrainingReport> {
        const modelRepresentation = this.getModelRepresentation();

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;
        let testAccuracy: Scalar | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );
        const trainAccuracy = accuracy(trainingData.y, yTraining!);

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );

            testAccuracy = accuracy(testData.y, yTesting!);
        }

        const [
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            trainAccuracyValue,
            // test
            testPredictedLabels,
            testAccuracyValue,
        ] = await Promise.all([
            getTensorArray(yPredictions),
            // train
            getTensorArray(yTraining),
            getTensorData(trainAccuracy),
            // test
            getTensorArray(yTesting),
            getTensorData(testAccuracy),
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
            taskType: 'classification',
            iterations: this.getIterations(),
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
        };
    }

    dispose(): void {
        this.thetaArray.forEach((theta) => theta.dispose());
    }
}
