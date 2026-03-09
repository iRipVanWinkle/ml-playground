import {
    type Tensor2D,
    type Scalar,
    tidy,
    concat,
    matMul,
    variable,
    keep,
    onesLike,
} from '@tensorflow/tfjs';
import { BaseEstimator, type ModelOptions } from '../base/BaseEstimator';
import type { PredictionMetadata } from '../../types';
import { assertModelTrained } from '../../utils';

type NeuralNetworkOptions = ModelOptions & {
    layers: Array<{
        units: number;
        activation?: string;
    }>;
};

export class NeuralNetwork extends BaseEstimator {
    private _initTheta: Tensor2D[] | null = null; // for testing purposes

    private layers: NeuralNetworkOptions['layers'];

    constructor(options: NeuralNetworkOptions) {
        super(options);

        if (options.layers.length === 0) {
            throw new Error('No layers defined in the neural network.');
        }

        this.layers = options.layers;
    }

    async train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D> {
        const asLogits = this.lossFunc.usesLogits?.();

        // Define the loss function
        const lossFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D): Scalar => {
            const thetas = this.unpackParameters(theta);
            // Compute the predictions using the hypothesis function
            const yPred = this.forwardPropagation(X, thetas, asLogits);
            // Compute the loss using the loss function
            const loss = this.lossFunc.compute(y, yPred); // MSE
            // Compute the regularization gradient
            const penalty = this.regularization.compute(theta);

            // Add the regularization gradient to the loss gradient
            return loss.add(penalty);
        };

        // Define the gradient function
        const gradientFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D): Tensor2D => {
            const thetas = this.unpackParameters(theta);

            const activations: Array<Tensor2D> = [];
            const preActivations: Array<Tensor2D> = [];

            this.forwardPropagation(X, thetas, false /* asLogits */, preActivations, activations);
            const grads = this.backwardPropagation(y, thetas, preActivations, activations);
            const gradTheta = this.packParameters(grads);

            activations.forEach((tensor) => tensor.dispose()); // Dispose of activations to free memory
            preActivations.forEach((tensor) => tensor.dispose()); // Dispose of pre-activations

            return gradTheta;
        };

        const initTheta = this.initTheta();

        // Optimize theta using the provided optimizer
        this.theta = await this.optimizer.optimize({
            X,
            y,
            lossFunction,
            gradientFunction,
            initTheta,
        });

        return this.theta;
    }

    predict(X: Tensor2D, theta?: Tensor2D): Tensor2D {
        const resolvedTheta = theta ?? this.theta;

        assertModelTrained(resolvedTheta);

        const result = tidy(() => {
            const unpackedTheta = this.unpackParameters(resolvedTheta);

            const rawOutput = this.forwardPropagation(X, unpackedTheta);

            return this.probabilityToClassIndex(rawOutput);
        });

        return result;
    }

    predictWithMetadata(X: Tensor2D, theta?: Tensor2D): PredictionMetadata {
        const resolvedTheta = theta ?? this.theta;

        assertModelTrained(resolvedTheta);

        const [probabilities, predictions] = tidy(() => {
            const unpackedTheta = this.unpackParameters(resolvedTheta);

            const probabilities = this.forwardPropagation(X, unpackedTheta);

            const predictions = this.probabilityToClassIndex(probabilities);

            return [probabilities, predictions];
        });

        const isClassification = this.isMultiClassClassification() || this.isBinaryClassification();

        return isClassification
            ? {
                  type: 'classification',
                  predictions,
                  probabilities,
                  dispose() {
                      predictions.dispose();
                      probabilities.dispose();
                  },
              }
            : {
                  type: 'regression',
                  predictions,
                  dispose() {
                      predictions.dispose();
                      probabilities.dispose();
                  },
              };
    }

    usesOneHotLabels(): boolean {
        return this.isMultiClassClassification();
    }

    private forwardPropagation(
        X: Tensor2D,
        thetas: Array<Tensor2D>,
        asLogits = false,
        preActivations?: Array<Tensor2D>,
        activations?: Array<Tensor2D>,
    ): Tensor2D {
        const numLayers = this.layers.length;
        let input = X.clone(); // for safty disposing in future

        if (Array.isArray(activations)) {
            const activation = input.clone(); // Clone to avoid disposing in future
            activations.push(activation);
        }

        for (let i = 0; i < numLayers - 1; i++) {
            const theta = thetas[i];
            const layer = this.layers[i + 1];
            const isLast = i === numLayers - 2;

            const output = tidy(() => {
                const features = this.addBiasTerm(input);
                // Compute the logit and apply activation
                const z = features.matMul(theta) as Tensor2D;
                const output = isLast && asLogits ? z : this.applyActivation(z, layer.activation);

                if (Array.isArray(preActivations)) {
                    preActivations.push(keep(z.clone())); // Clone to avoid disposing in future
                }

                if (Array.isArray(activations)) {
                    const activation = output.clone(); // Clone to avoid disposing in future
                    activations.push(keep(activation));
                }

                return output;
            });

            input.dispose();

            input = output; // Update input for the next layer
        }

        return input; // Return the final output tensor
    }

    private backwardPropagation(
        y: Tensor2D,
        thetas: Array<Tensor2D>,
        preActivations: Array<Tensor2D>,
        activations: Array<Tensor2D>,
    ): Array<Tensor2D> {
        const grads = [];

        if (preActivations.length === 0 || activations.length === 0) {
            throw new Error('No pre-activations or activations provided for backpropagation.');
        }

        let layer = this.layers.at(-1);

        const dLoss = this.lossFunc.predictionGradient(y, activations.at(-1)!); // ∂L/∂ŷ
        const dActivation = this.applyActivationDeriv(preActivations.at(-1)!, layer?.activation); // ∂ŷ/∂z
        let delta = dLoss.mul(dActivation); // ∂L/∂z = ∂L/∂ŷ ⋅ ∂ŷ/∂z

        for (let l = thetas.length - 1; l >= 0; l--) {
            layer = this.layers[l]; // Get the current layer
            const aPrevWithBias = this.addBiasTerm(activations[l]); // Add bias term to the previous activation
            const grad = matMul(aPrevWithBias.transpose(), delta).div(delta.shape[0]);
            grads[l] = grad as Tensor2D;

            if (l > 0) {
                const W_noBias = thetas[l].slice([1, 0], [-1, -1]).transpose(); // remove bias row, then transpose
                const da = matMul(delta, W_noBias);
                delta = da.mul(
                    this.applyActivationDeriv(preActivations[l - 1]!, layer?.activation),
                );
            }
        }

        return grads;
    }

    private applyActivation(X: Tensor2D, activation?: string): Tensor2D {
        switch (activation) {
            case undefined:
            case 'linear':
                // Linear activation: return input as is
                return X;
            case 'relu':
                // ReLU activation: apply ReLU function
                return X.relu();
            case 'sigmoid':
                // Sigmoid activation: apply sigmoid function
                return X.sigmoid();
            case 'tanh':
                // Tanh activation: apply tanh function
                return X.tanh();
            case 'softmax':
                // Softmax activation: apply softmax function
                return X.softmax();
            case 'leakyReLU':
                // Leaky ReLU activation: apply leaky ReLU function
                return X.leakyRelu(0.01);
            case 'swish':
                // Swish activation: apply swish function
                return X.mul(X.sigmoid());
            case 'softplus':
                // Softplus activation: apply softplus function
                return X.softplus();
            default:
                throw new Error(`Unsupported activation function: ${activation}`);
        }
    }

    private applyActivationDeriv(X: Tensor2D, activation?: string): Tensor2D {
        switch (activation) {
            case undefined:
            case 'linear':
                // Derivative of linear activation is 1
                return X.onesLike();
            case 'relu':
                // Derivative of ReLU: 1 for positive inputs, 0 for negative inputs
                return X.greater(0).toFloat() as Tensor2D;
            case 'sigmoid': {
                // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
                const sig = X.sigmoid();
                return sig.mul(sig.neg().add(1));
            }
            case 'tanh': {
                // Derivative of tanh: 1 - tanh^2(x)
                const tanh = X.tanh();
                return tanh.neg().add(1).mul(tanh);
            }
            case 'softmax':
                // Return ones because the derivative is not used directly
                return onesLike(X);
            case 'leakyReLU':
                // Derivative of leaky ReLU: 1 for positive inputs, alpha for negative inputs
                return X.greater(0).toFloat().add(X.lessEqual(0).toFloat().mul(0.01));
            case 'swish': {
                // Derivative of swish: swish(x) + sigmoid(x) * (1 - swish(x))
                const swish = X.mul(X.sigmoid());
                return swish.add(X.sigmoid().mul(X.neg().add(1)));
            }
            case 'softplus':
                // Derivative of softplus: sigmoid(x)
                return X.sigmoid();
            default:
                throw new Error(`Unsupported activation function: ${activation}`);
        }
    }

    private packParameters(params: Tensor2D[]): Tensor2D {
        const allParams = [];

        for (let i = 0; i < params.length; i++) {
            allParams.push(params[i].flatten());
        }

        return concat(allParams).reshape([-1, 1]) as Tensor2D;
    }

    private unpackParameters(params: Tensor2D): Array<Tensor2D> {
        const unpackedParams: Tensor2D[] = [];
        let offset = 0;

        for (let i = 0; i < this.layers.length - 1; i++) {
            const inSize = this.layers[i].units + 1;
            const outSize = this.layers[i + 1].units;
            const size = inSize * outSize;

            const flat = params
                .slice([offset, 0], [size, 1])
                .reshape([inSize, outSize]) as Tensor2D;
            offset += size;

            unpackedParams.push(flat);
        }

        return unpackedParams;
    }

    private isBinaryClassification(): boolean {
        return this.layers.at(-1)?.activation === 'sigmoid';
    }

    private isMultiClassClassification(): boolean {
        return this.layers.at(-1)?.activation === 'softmax';
    }

    private probabilityToClassIndex(probability: Tensor2D): Tensor2D {
        return tidy(() => {
            if (this.isMultiClassClassification()) {
                // Find the indices of the maximum probabilities
                const maxIndices = probability.argMax(1);

                return maxIndices.reshape([-1, 1]) as Tensor2D;
            }

            if (this.isBinaryClassification()) {
                // Threshold at 0.5: returns 1 if probability >= 0.5, else 0
                return probability.greaterEqual(0.5).cast('float32');
            }

            return probability; // For regression or other cases, return raw probabilities
        });
    }

    private initTheta(): Tensor2D {
        if (this._initTheta) {
            return this.packParameters(this._initTheta);
        }

        const layerSizes = this.layers.map((layer) => layer.units);
        const weights: Tensor2D[] = [];
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const shape: [number, number] = [layerSizes[i], layerSizes[i + 1]];
            const theta = this.thetaInitializer(shape);

            weights.push(theta);
        }

        const theta = variable(this.packParameters(weights));
        theta.print();
        weights.forEach((tensor) => tensor.dispose()); // Dispose of the weights to free memory

        return theta;
    }
}
