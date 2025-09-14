import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { BatchGD } from '../../optimizers/batch';
import { NeuralNetwork } from './NeuralNetwork';
import type { LossFunction } from '../../types';
import {
    BinaryCrossentropy,
    CategoricalCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
} from '../../losses';

const round = (array: number[] | number[][] | number[][][]): number[] => {
    return array.map((row) =>
        Array.isArray(row) ? round(row) : Number(row.toFixed(4)),
    ) as number[];
};

const inithThetaFunction = (layers: { units: number }[]) => {
    const layerSizes = layers.map((layer) => layer.units);
    const weights: tf.Tensor2D[] = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
        // Add 1 for bias
        const shape: [number, number] = [layerSizes[i] + 1, layerSizes[i + 1]];
        // He/Xavier initialization
        const w = tf.randomNormal(
            shape,
            0,
            Math.sqrt(2 / (layerSizes[i] + layerSizes[i + 1])),
            'float32',
            42,
        );
        weights.push(w as tf.Tensor2D);
    }

    return weights;
};

export function createTFE(
    lossFn: LossFunction,
    layers: Array<{
        units: number;
        activation?: string;
    }>,
    thetas: tf.Tensor2D[],
    learningRate = 0.1,
) {
    // Create sequential model
    const model = tf.sequential();

    // Add layers with proper activations
    for (let i = 1; i < layers.length; i++) {
        const layer = layers[i];

        model.add(
            tf.layers.dense({
                inputShape: i === 1 ? [layers[i - 1].units] : undefined, // Input shape for the first layer
                units: layer.units, // Output units
                activation: layer.activation as undefined, // No activation on output for regression
            }),
        );
    }

    // Convert your theta format to TensorFlow.js format
    const weightsAndBiases = thetas.map((theta) => {
        const biases = theta.slice([0, 0], [1, -1]).reshape([-1]);
        const weights = theta.slice([1, 0], [-1, -1]);
        return [weights, biases];
    });
    model.setWeights(weightsAndBiases.flat());

    function computeLoss(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor {
        const result = lossFn.compute(yTrue as tf.Tensor2D, yPred as tf.Tensor2D);

        return result;
    }

    // Compile with MSE loss
    model.compile({
        optimizer: tf.train.sgd(learningRate),
        loss: computeLoss,
    });

    return {
        model,
        train: async function (X: tf.Tensor2D, y: tf.Tensor2D, iterations = 1000) {
            await model.fit(X, y, {
                epochs: iterations,
                batchSize: X.shape[0],
                verbose: 0, // Silent training
            });

            // Convert back to theta format
            const newWeights = model.getWeights();
            const newThetas: tf.Tensor2D[] = [];
            for (let i = 0; i < newWeights.length; i += 2) {
                const weights = newWeights[i] as tf.Tensor2D;
                const biases = newWeights[i + 1] as tf.Tensor1D;

                const biasRow = biases.reshape([1, -1]);
                const theta = tf.concat([biasRow, weights], 0) as tf.Tensor2D;
                newThetas.push(theta);
            }

            return newThetas;
        },
    };
}

describe('NeuralNetwork', () => {
    let X: tf.Tensor2D;
    let lossFunc: LossFunction;
    let optimizer: BatchGD;
    let model: NeuralNetwork;

    beforeEach(() => {
        lossFunc = new BinaryCrossentropy();
        optimizer = new BatchGD({ learningRate: 0.01, maxIterations: 2, tolerance: 0.001 });

        model = new NeuralNetwork({
            lossFunc,
            optimizer,
            layers: [{ units: 1, activation: 'linear' }],
        });
    });

    afterEach(() => {
        model.dispose?.();
    });

    describe('::packParameters', () => {
        it('should correctly pack parameters into a single tensor', async () => {
            const theta1 = tf.tensor2d([
                [0.0, 0.0], // bias
                [0.5, -0.4], // weights from feature 1
                [0.3, 0.2], // weights from feature 2
            ]);
            const theta2 = tf.tensor2d([
                [0.0], // bias
                [0.6], // weights from feature 1
                [-0.5], // weights from feature 2
            ]);
            const thetas = [theta1, theta2];

            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 2, activation: 'linear' },
                { units: 1, activation: 'linear' }, // output layer
            ];
            model = new NeuralNetwork({ lossFunc, optimizer, layers });

            const packedThetas = model['packParameters'](thetas);

            expect(round(packedThetas.arraySync())).toEqual([
                [0.0],
                [0.0],
                [0.5],
                [-0.4],
                [0.3],
                [0.2],
                [0.0],
                [0.6],
                [-0.5],
            ]);
        });
    });

    describe('::unpackParameters', () => {
        it('should correctly unpack parameters from a single tensor', async () => {
            const packedThetas = tf.tensor2d([
                [0.0],
                [0.0],
                [0.5],
                [-0.4],
                [0.3],
                [0.2],
                [0.0],
                [0.6],
                [-0.5],
            ]);

            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 2, activation: 'linear' },
                { units: 1, activation: 'linear' }, // output layer
            ];
            model = new NeuralNetwork({ lossFunc, optimizer, layers });

            const thetas = model['unpackParameters'](packedThetas);
            expect(thetas.map((theta) => round(theta.arraySync()))).toEqual([
                [
                    [0.0, 0.0], // bias
                    [0.5, -0.4], // weights from feature 1
                    [0.3, 0.2], // weights from feature 2
                ],
                [
                    [0.0], // bias
                    [0.6], // weights from feature 1
                    [-0.5], // weights from feature 2
                ],
            ]);
        });
    });

    describe('::predict', () => {
        it('should make predictions with 2 features and predefined theta', async () => {
            X = tf.tensor2d([
                [1, 2],
                [2, 1],
                [3, 5],
                [0, 0],
                [-1, 2],
            ]);
            const theta = tf.tensor2d([[2.0], [3.0], [-1.0]]);

            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 1, activation: 'linear' },
            ];
            model = new NeuralNetwork({ lossFunc, optimizer, layers });

            const prevNumTensors = tf.memory().numTensors;

            const prediction = model.predict(X, theta);

            const expectedNumTensors = prevNumTensors + 1; // +1 for the prediction tensor

            expect(prediction.arraySync()).toEqual([[3], [7], [6], [2], [-3]]);
            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });

    describe('::forwardPropagation', () => {
        it('should perform forward propagation with preActivations and activations', async () => {
            X = tf.tensor2d([
                [1, 2],
                [2, 1],
                [3, 5],
                [0, 0],
                [-1, 2],
            ]);
            const theta = tf.tensor2d([[2.0], [3.0], [-1.0]]);

            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 1, activation: 'linear' },
            ];
            model = new NeuralNetwork({ lossFunc, optimizer, layers });
            const unpackedTheta = model['unpackParameters'](theta);

            const prevNumTensors = tf.memory().numTensors;

            const preActivations = [] as tf.Tensor2D[];
            const activations = [] as tf.Tensor2D[];

            const prediction = model['forwardPropagation'](
                X,
                unpackedTheta,
                false,
                preActivations,
                activations,
            );

            const expectedNumTensors = prevNumTensors + 1 + 1 + 2; // +1 for the prediction tensor, +1 for preActivations, +2 for activations

            expect(preActivations.length).toEqual(1);
            expect(activations.length).toEqual(2);
            expect(prediction.arraySync()).toEqual([[3], [7], [6], [2], [-3]]);
            expect(tf.memory().numTensors).toEqual(expectedNumTensors);
        });
    });

    describe('::train', () => {
        it('should train with sigmoid activation', async () => {
            const learningRate = 0.1;
            const maxIterations = 100;
            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 4, activation: 'sigmoid' },
                { units: 3, activation: 'sigmoid' },
                { units: 1, activation: 'sigmoid' },
            ];
            const X = tf.tensor2d([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]);
            const y = tf.tensor2d([[0], [1], [1], [0]]);
            const initThetas = inithThetaFunction(layers);

            // Initialize the model with the specified layers and loss function
            optimizer = new BatchGD({ learningRate, maxIterations });
            lossFunc = new MeanSquaredError();

            // Create the TFE model with the specified layers and initial thetas
            const tfThetas = await createTFE(lossFunc, layers, initThetas, learningRate).train(
                X,
                y,
                maxIterations,
            );
            const tfThetasArr = tfThetas.map((t) => t.arraySync());

            model = new NeuralNetwork({ lossFunc, optimizer, layers });
            model['_initTheta'] = initThetas;

            const theta = await model.train(X, y);

            const thetas = model['unpackParameters'](theta);
            const thetasArr = thetas.map((t) => t.arraySync());

            expect(round(tfThetasArr)).toEqual(round(thetasArr));
        });

        it('should train with linear activation', async () => {
            const learningRate = 0.1;
            const maxIterations = 100;
            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 3, activation: 'linear' },
                { units: 1, activation: 'linear' },
            ];
            const X = tf.tensor2d([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]);
            const y = tf.tensor2d([[0], [1], [1], [0]]);
            const initThetas = inithThetaFunction(layers);

            // Initialize the model with the specified layers and loss function
            optimizer = new BatchGD({ learningRate, maxIterations });
            lossFunc = new MeanSquaredError();

            // Create the TFE model with the specified layers and initial thetas
            const tfThetas = await createTFE(lossFunc, layers, initThetas, learningRate).train(
                X,
                y,
                maxIterations,
            );
            const tfThetasArr = tfThetas.map((t) => t.arraySync());

            model = new NeuralNetwork({ lossFunc, optimizer, layers });
            model['_initTheta'] = initThetas;

            const theta = await model.train(X, y);

            const thetas = model['unpackParameters'](theta);
            const thetasArr = thetas.map((t) => t.arraySync());

            expect(round(tfThetasArr)).toEqual(round(thetasArr));
        });

        it('should train with relu activation', async () => {
            const learningRate = 0.1;
            const maxIterations = 100;
            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 4, activation: 'relu' },
                { units: 3, activation: 'relu' },
                { units: 1, activation: 'relu' },
            ];
            const X = tf.tensor2d([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]);
            const y = tf.tensor2d([[0], [1], [1], [0]]);
            const initThetas = inithThetaFunction(layers);

            // Initialize the model with the specified layers and loss function
            optimizer = new BatchGD({ learningRate, maxIterations });
            lossFunc = new MeanAbsoluteError();

            // Create the TFE model with the specified layers and initial thetas
            const tfThetas = await createTFE(lossFunc, layers, initThetas, learningRate).train(
                X,
                y,
                maxIterations,
            );
            const tfThetasArr = tfThetas.map((t) => t.arraySync());

            model = new NeuralNetwork({ lossFunc, optimizer, layers });
            model['_initTheta'] = initThetas;

            const theta = await model.train(X, y);

            const thetas = model['unpackParameters'](theta);
            const thetasArr = thetas.map((t) => t.arraySync());

            expect(round(tfThetasArr)).toEqual(round(thetasArr));
        });

        it('should train with softmax and relu activation', async () => {
            const learningRate = 0.1;
            const maxIterations = 100;
            const layers = [
                { units: 2 }, // input layer with 2 features
                { units: 8, activation: 'relu' }, // hidden layer 1
                { units: 4, activation: 'relu' }, // hidden layer 2
                { units: 3, activation: 'softmax' },
            ];
            const X = tf.tensor2d([
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
                [0.5, 0.5],
                [0.3, 0.7],
            ]);
            const y = tf.tensor2d([
                [1, 0, 0], // Class A
                [0, 1, 0], // Class B
                [0, 0, 1], // Class C
                [1, 0, 0], // Class A
                [0, 1, 0], // Class B
                [0, 0, 1], // Class C
            ]);

            const initThetas = inithThetaFunction(layers);

            // Initialize the model with the specified layers and loss function
            optimizer = new BatchGD({ learningRate, maxIterations });
            lossFunc = new CategoricalCrossentropy();

            // Create the TFE model with the specified layers and initial thetas
            const tfThetas = await createTFE(lossFunc, layers, initThetas, learningRate).train(
                X,
                y,
                maxIterations,
            );
            const tfThetasArr = tfThetas.map((t) => t.arraySync());

            model = new NeuralNetwork({ lossFunc, optimizer, layers });
            model['_initTheta'] = initThetas;

            const theta = await model.train(X, y);

            const thetas = model['unpackParameters'](theta);
            const thetasArr = thetas.map((t) => t.arraySync());

            expect(round(tfThetasArr)).toEqual(round(thetasArr));
        });
    });
});
