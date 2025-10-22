import { type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, ModelRepresentation, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { LinearTrainingReport } from '../types';

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

export class LinearLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, LinearTrainingReport>
{
    private lossHistory: number[] = [];
    private iterationCount: number = 0;
    private theta?: Tensor2D;

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

        this.theta = theta;
        this.lossHistory.push(loss);

        this.iterationCount = iteration + 1;
    }

    getModelRepresentation(): Tensor2D {
        return this.theta!.clone();
    }

    async calculateMetrics(): Promise<LinearTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const theta = this.theta!;

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, theta);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            theta,
        );

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                theta,
            );
        }

        const [
            thetaArray,
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            // test
            testPredictedLabels,
            testLossValue,
        ] = await Promise.all([
            getTensorArray(theta),
            getTensorArray(yPredictions),
            // train
            getTensorArray(yTraining),
            // test
            getTensorArray(yTesting),
            getTensorData(testLoss),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        yTraining?.dispose();
        yTestingProbability?.dispose();
        yTesting?.dispose();
        yTrainingProbability?.dispose();
        trainLoss?.dispose();
        testLoss?.dispose();

        return {
            type: 'linear',
            taskType: 'regression',
            trainLossHistory: [this.lossHistory],
            iteration: this.iterationCount,
            trainLoss: this.lossHistory.at(-1) ?? 0,
            testLoss: testLossValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            theta: thetaArray!,
        };
    }

    dispose(): void {
        this.theta?.dispose();
    }
}
