import { concat, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, OptimizerCallbackParameters } from '@/ml/types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import type { LogisticTrainingReport } from '../types';
import {
    getSafeMatrixFromTensor,
    getSafeTensorArray,
    getSafeTensorValue,
} from '@/app/shared/workers';
import { accuracy, confusionMatrix } from '@/ml/metrics';
import { confusionMatrixData } from '@/app/shared/visualization/metrics/confusion-matrix/calculations';
import { rocCurveData } from '@/app/shared/visualization/plots/roc-curve/calculations';

function fixLength(matrix: number[][]): number[][] {
    const minLength = Math.min(...matrix.map((m) => m.length)) ?? 0;
    return matrix.map((m) => m.slice(0, minLength));
}

export class LogisticLiveMetrics
    implements LiveMetrics<OptimizerCallbackParameters, LogisticTrainingReport>
{
    private lossHistory: number[][] = [];
    private iterationCounts: number[] = [];
    private thetaArray: Tensor2D[] = [];

    private model: Model<Tensor2D>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new LogisticLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: OptimizerCallbackParameters): void {
        const { threadId, iteration, theta, loss } = params;

        this.thetaArray[threadId] = theta;

        for (let i = 0; i <= threadId; i++) {
            this.lossHistory[i] = this.lossHistory[i] ?? [];
        }
        this.lossHistory[threadId].push(loss);

        for (let i = 0; i <= threadId; i++) {
            this.iterationCounts[i] = this.iterationCounts[i] ?? 0;
        }
        this.iterationCounts[threadId] = iteration + 1;
    }

    getIterations(): number[] {
        return [...this.iterationCounts];
    }

    getModelRepresentation(): Tensor2D {
        return concat(this.thetaArray.filter(Boolean), 1) as Tensor2D;
    }

    getFormattedLossHistory(): number[][] {
        return fixLength(this.lossHistory);
    }

    async calculateMetrics(): Promise<LogisticTrainingReport> {
        const modelRepresentation = this.getModelRepresentation();

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();
        const numClasses = this.datasetManager.getNumClasses();

        let yPredictions: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let testLoss: Scalar | undefined;
        let testAccuracy: Scalar | undefined;
        let testConfusionMatrix: Tensor2D | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, modelRepresentation);
        }

        const [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            modelRepresentation,
        );
        const trainAccuracy = accuracy(trainingData.y, yTraining!);
        const trainConfusionMatrix = confusionMatrix(trainingData.y, yTraining, numClasses);

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                modelRepresentation,
            );

            testAccuracy = accuracy(testData.y, yTesting!);
            testConfusionMatrix = confusionMatrix(testData.y, yTesting!, numClasses);
        }

        const [
            thetaArray,
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            trainAccuracyValue,
            trainConfusionMatrixValue,
            trainProbabilityValue,
            trainLabelValue,
            // test
            testPredictedLabels,
            testAccuracyValue,
            testConfusionMatrixValue,
            testProbabilityValue,
            testLabelValue,
        ] = await Promise.all([
            getSafeMatrixFromTensor(modelRepresentation),
            getSafeMatrixFromTensor(yPredictions),
            // train
            getSafeMatrixFromTensor(yTraining),
            getSafeTensorValue(trainAccuracy),
            getSafeTensorArray(trainConfusionMatrix),

            getSafeMatrixFromTensor(yTrainingProbability),
            getSafeMatrixFromTensor(trainingData.y),
            // test
            getSafeMatrixFromTensor(yTesting),
            getSafeTensorValue(testAccuracy),
            getSafeTensorArray(testConfusionMatrix),
            getSafeMatrixFromTensor(yTestingProbability),
            getSafeMatrixFromTensor(testData?.y),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        yTraining?.dispose();
        yTestingProbability?.dispose();
        yTesting?.dispose();
        yTrainingProbability?.dispose();
        trainLoss?.dispose();
        testLoss?.dispose();
        trainConfusionMatrix?.dispose();
        testConfusionMatrix?.dispose();
        // modelRepresentation.dispose();

        return {
            type: 'logistic',
            taskType: 'classification',
            trainLossHistory: this.getFormattedLossHistory(),
            iterations: this.getIterations(),
            testAccuracy: testAccuracyValue!,
            trainAccuracy: trainAccuracyValue!,
            trainPredictedLabels: trainPredictedLabels!,
            testPredictedLabels: testPredictedLabels!,
            predictionPredictedLabels: predictionPredictedLabels,
            theta: thetaArray!,
            trainConfusionMatrix: confusionMatrixData(trainConfusionMatrixValue, numClasses),
            testConfusionMatrix: testConfusionMatrixValue
                ? confusionMatrixData(testConfusionMatrixValue, numClasses)
                : undefined,

            trainRocCurve: rocCurveData(
                trainLabelValue,
                trainProbabilityValue,
                trainConfusionMatrixValue,
            ),
            testRocCurve:
                testLabelValue && testProbabilityValue && testConfusionMatrixValue
                    ? rocCurveData(testLabelValue, testProbabilityValue, testConfusionMatrixValue)
                    : undefined,
        };
    }

    dispose(): void {
        this.thetaArray.forEach((theta) => theta.dispose());
    }
}
