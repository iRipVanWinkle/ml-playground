import { type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { NeuralClassificationTrainingReport } from '../types';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';

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

export class NeuralClassificationLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, NeuralClassificationTrainingReport>
{
    private lossHistory: number[] = [];
    private iterationCount: number = 0;
    private theta?: Tensor2D;

    private model: Model<Tensor2D>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new NeuralClassificationLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: OptimizerCallbackParameters): void {
        const { iteration, theta, loss } = params;

        this.theta = theta;

        this.lossHistory = this.lossHistory ?? [];
        this.lossHistory.push(loss);

        this.iterationCount = iteration + 1;
    }

    async calculateMetrics(): Promise<NeuralClassificationTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const theta = this.theta!;

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;
        let testAccuracy: Scalar | undefined;
        let testConfusionMatrix: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, theta);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            theta,
        );
        const trainAccuracy = accuracy(trainingData.y, yTraining!);
        const trainConfusionMatrix = confusionMatrix(
            trainingData.y,
            yTraining,
            this.datasetManager.getNumClasses(),
        );
        trainAccuracy.print();
        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                theta,
            );

            testAccuracy = accuracy(testData.y, yTesting!);
            testConfusionMatrix = confusionMatrix(
                testData.y,
                yTesting!,
                this.datasetManager.getNumClasses(),
            );
        }

        const [
            thetaArray,
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            trainAccuracyValue,
            trainConfusionMatrixValue,
            // test
            testPredictedLabels,
            testAccuracyValue,
            testConfusionMatrixValue,
        ] = await Promise.all([
            getTensorArray(theta),
            getTensorArray(yPredictions),
            // train
            getTensorArray(yTraining),
            getTensorData(trainAccuracy),
            getTensorArray(trainConfusionMatrix),
            // test
            getTensorArray(yTesting),
            getTensorData(testAccuracy),
            getTensorArray(testConfusionMatrix),
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
            type: 'neural',
            taskType: 'classification',
            trainLossHistory: [this.lossHistory],
            iteration: this.iterationCount,
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            theta: thetaArray!,
            trainConfusionMatrix: confusionMatrixData(
                trainConfusionMatrixValue!,
                this.datasetManager.getNumClasses(),
            ),
            testConfusionMatrix: testConfusionMatrixValue
                ? confusionMatrixData(testConfusionMatrixValue, this.datasetManager.getNumClasses())
                : undefined,
        };
    }

    dispose(): void {
        this.theta?.dispose();
    }
}
