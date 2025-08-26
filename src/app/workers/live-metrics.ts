import { Tensor, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import type { Model, ModelRepresentation } from '@/ml/types';
import { accuracy } from '@/ml/metrics';
import type { DatasetManager } from './dataset-manager';
import type { LiveMetricsProps } from './live-metrics-props';
import type { TrainingSession } from './training-session';

export type LiveResults = {
    trainPredictedLabels?: number[][];
    testPredictedLabels?: number[][];
    predictionPredictedLabels?: number[][];
    trainAccuracy?: number;
    testAccuracy?: number;
    trainLoss?: number;
    testLoss?: number;
    thetaArray?: number[][];
};

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

export class LiveMetrics {
    private model: Model<ModelRepresentation>;
    private datasetManager: DatasetManager;
    private props: LiveMetricsProps;

    constructor(
        model: Model<ModelRepresentation>,
        datasetManager: DatasetManager,
        props: LiveMetricsProps,
    ) {
        this.model = model;
        this.datasetManager = datasetManager;
        this.props = props;
    }

    async calculate(session: TrainingSession): Promise<LiveResults> {
        const combinedTheta = session.getCombinedTheta();

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        let yPredictions: Tensor2D | undefined;
        let yTraining: Tensor2D | undefined;
        let yTrainingProbability: Tensor2D | undefined;
        let yTesting: Tensor2D | undefined;
        let yTestingProbability: Tensor2D | undefined;
        let trainAccuracy: Scalar | undefined;
        let testAccuracy: Scalar | undefined;
        let trainLoss: Scalar | undefined;
        let testLoss: Scalar | undefined;

        if (predictionData) {
            yPredictions = this.model.predict(predictionData, combinedTheta);
        }

        const metrics = this.props.isClassificationTask ? [accuracy] : [];
        // eslint-disable-next-line prefer-const
        [yTraining, yTrainingProbability, trainLoss] = this.model.evaluate(
            trainingData.X,
            trainingData.y,
            combinedTheta,
        );
        // eslint-disable-next-line prefer-const
        [trainAccuracy] = metrics.map((metric) => metric(trainingData.y, yTraining!));

        if (testData) {
            [yTesting, yTestingProbability, testLoss] = this.model.evaluate(
                testData.X,
                testData.y,
                combinedTheta,
            );
            [testAccuracy] = metrics.map((metric) => metric(testData.y, yTesting!));
        }

        const [
            thetaArray,
            predictionPredictedLabels,
            trainPredictedLabels,
            testPredictedLabels,
            trainAccuracyValue,
            testAccuracyValue,
            testLossValue,
            trainLossValue,
        ] = await Promise.all([
            getTensorArray(combinedTheta instanceof Tensor ? combinedTheta : undefined, []),
            getTensorArray(yPredictions),
            getTensorArray(yTraining),
            getTensorArray(yTesting),
            getTensorData(trainAccuracy),
            getTensorData(testAccuracy),
            getTensorData(testLoss),
            getTensorData(trainLoss),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        yTraining?.dispose();
        yTestingProbability?.dispose();
        yTesting?.dispose();
        yTrainingProbability?.dispose();
        trainAccuracy?.dispose();
        testAccuracy?.dispose();
        trainLoss?.dispose();
        testLoss?.dispose();
        if (combinedTheta instanceof Tensor) {
            combinedTheta.dispose();
        }

        return {
            trainPredictedLabels,
            testPredictedLabels,
            predictionPredictedLabels,
            trainAccuracy: trainAccuracyValue,
            testAccuracy: testAccuracyValue,
            trainLoss: trainLossValue,
            testLoss: testLossValue,
            thetaArray,
        };
    }
}
