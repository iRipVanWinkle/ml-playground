import { createModel } from '@/app/helpers/createModel';
import { encode } from '@/app/helpers/float32Array';
import { accuracy } from '@/ml/metrics';
import type { Model } from '@/ml/types';
import type { State } from '@/app/store';
import { Tensor, memory, tensor2d, type Tensor2D, concat, type Scalar } from '@tensorflow/tfjs';

type TrainingCallbacks = {
    onReport: (report: Float32Array) => void;
    onInfo: (msg: string) => void;
    onError: (msg: string) => void;
    onFinished: () => void;
};

export interface TrainerReport {
    trainLossHistory: number[][];
    trainAccuracy: number;
    testAccuracy: number;
    testLoss: number;
    iterations: number[];
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    predictionPredictedLabels: number[][];
    theta: number[][];
}

function fixLength(matrix: number[][]): number[][] {
    const minLength = Math.min(...matrix.map((m) => m.length));
    return matrix.map((m) => m.slice(0, minLength));
}

function tensor2dIfPopulated(data?: number[][]): Tensor2D | undefined {
    if (!data || data.length === 0) return undefined;

    return tensor2d(data);
}

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

export class Trainer {
    private model: Model | null = null;

    async train(state: State, byStep: boolean, callbacks: TrainingCallbacks) {
        console.info('Start', memory());

        const { modelSettings, dataSettings, taskType, data } = state;

        const X = tensor2d(data.trainInputFeatures);
        const y = tensor2d(data.trainTargetLabels);
        const XTest = tensor2dIfPopulated(data.testInputFeatures);
        const yTest = tensor2dIfPopulated(data.testTargetLabels);
        const XPredictions = tensor2dIfPopulated(data.predictionInputFeatures);

        const [model, optimizer] = createModel(modelSettings, dataSettings);
        this.model = model;

        const thetasArray: Tensor2D[] = [];

        const numThreads = 1;

        const lossArray: number[] = Array.from({ length: numThreads }, () => 0);
        const lossHistoryArray: number[][] = Array.from({ length: numThreads }, () => []);
        const iterations: number[] = Array.from({ length: numThreads }, () => 0);

        optimizer.on('error', (message) => {
            console.error(message);
            callbacks.onError(message);

            model.stop();
        });

        optimizer.on('info', (message) => {
            console.info(message);
            callbacks.onInfo(message);
        });

        optimizer.on('callback', async ({ threadId, iteration, theta }) => {
            const index = threadId;

            if (thetasArray[index] === undefined) {
                thetasArray[index] = theta;
            }

            const thetas = concat(thetasArray.filter(Boolean), 1);

            let yPredictions: Tensor2D | undefined;
            let yTraining: Tensor2D | undefined;
            let yTrainingProbability: Tensor2D | undefined;
            let yTesting: Tensor2D | undefined;
            let yTestingProbability: Tensor2D | undefined;
            let trainAccuracy: Scalar | undefined;
            let testAccuracy: Scalar | undefined;
            let trainLoss: Scalar | undefined;
            let testLoss: Scalar | undefined;

            if (XPredictions) {
                yPredictions = model.predict(XPredictions, thetas);
            }

            const metrics = taskType === 'classification' ? [accuracy] : [];
            // eslint-disable-next-line prefer-const
            [yTraining, yTrainingProbability, trainLoss] = model.evaluate(X, y, thetas);
            // eslint-disable-next-line prefer-const
            [trainAccuracy] = metrics.map((metric) => metric(y, yTraining!));

            if (XTest && yTest) {
                [yTesting, yTestingProbability, testLoss] = model.evaluate(XTest, yTest, thetas);
                [testAccuracy] = metrics.map((metric) => metric(yTest, yTesting!));
            }

            const [
                thetaArray,
                predictionLabels,
                yTrainingArray,
                yTestingArray,
                trainAccuracyValue,
                testAccuracyValue,
                testLossValue,
                trainLossValue,
            ] = await Promise.all([
                getTensorArray(thetas instanceof Tensor ? thetas : undefined, []),
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
            if (thetas instanceof Tensor) {
                thetas.dispose();
            }

            lossArray[index] = trainLossValue!;

            lossHistoryArray[index].push(trainLossValue!);
            iterations[index] = iteration + 1;

            const report = encode({
                trainLossHistory: fixLength(lossHistoryArray), // Ensure all loss history arrays are of the same length
                trainAccuracy: trainAccuracyValue,
                testAccuracy: testAccuracyValue,
                testLoss: testLossValue,
                iterations: iterations,
                trainPredictedLabels: yTrainingArray,
                testPredictedLabels: yTestingArray,
                predictionPredictedLabels: predictionLabels,
                theta: thetaArray,
            });

            callbacks.onReport(report);

            if (byStep && iteration === 0) {
                model.stop();
            }
        });

        console.time('Training Linear Regression Model');

        const theta = await model.train(X, y);

        /**
         * Theta
         * Linear Regression
         * [
         *  [... bias ...],
         *  [... waight1 ...],
         *  [... waight2 ...]
         * ]
         *
         * Logistic Regression
         * [     cat1, cat2
         *  [... bias, bias ...],
         *  [... waight1, waight1 ...],
         *  [... waight2, waight2 ...]
         * ]
         *
         */

        console.timeEnd('Training Linear Regression Model');

        X.dispose();
        y.dispose();
        XTest?.dispose();
        yTest?.dispose();
        XPredictions?.dispose();

        if (theta instanceof Tensor) {
            theta.dispose();
        }

        if (thetasArray instanceof Tensor) {
            thetasArray.forEach((t) => t.dispose());
        }

        model.dispose(true);
        this.model = null;

        console.info('End', memory());

        callbacks.onFinished?.();
    }

    stop() {
        this.model?.stop();
        this.model?.dispose(true);
        this.model = null;
    }

    pause() {
        this.model?.pause();
    }

    resume() {
        this.model?.resume();
    }

    step() {
        this.model?.step();
    }
}
