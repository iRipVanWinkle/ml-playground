import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { LogisticRegressor } from './LogisticRegressor';
import { assertModelTrained } from '../../utils';
import type { PredictionMetadata } from '../../types';

export class SoftmaxLogisticRegressor extends LogisticRegressor {
    private _initTheta: Tensor2D | null = null; // for testing purposes

    async train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D> {
        const numFeatures = X.shape[1];
        const numClasses = y.shape[1];
        const asLogits = this.lossFunc.usesLogits?.();

        const initTheta = this._initTheta ?? this.thetaInitializer([numFeatures, numClasses]);

        // Define the loss function
        const lossFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D): Scalar => {
            // Compute the predictions using the hypothesis function
            const yPred = this.hypothesis(X, theta, asLogits);
            // Compute the loss using the loss function
            const loss = this.lossFunc.compute(y, yPred);
            // Compute the regularization gradient
            const penalty = this.regularization.compute(theta);

            // Add the regularization gradient to the loss gradient
            return loss.add(penalty);
        };

        // Define the gradient function
        const gradientFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D): Tensor2D => {
            // Compute the predictions using the hypothesis function
            const yPred = this.hypothesis(X, theta);
            // Compute the gradients using the loss function
            const gradient = this.lossFunc.parameterGradient(X, y, yPred);
            // Compute the regularization gradient ([0, penalty, penalty, ..., penalty])
            const penalty = this.regularization.gradient(theta);

            // Add the regularization gradient to the loss gradient
            return gradient.add(penalty);
        };

        this.theta = await this.optimizer.optimize({
            X,
            y,
            lossFunction,
            gradientFunction,
            initTheta,
        });

        initTheta.dispose();

        return this.theta;
    }

    predict(X: Tensor2D, theta?: Tensor2D): Tensor2D {
        const resolvedTheta = theta ?? this.theta;

        assertModelTrained(resolvedTheta);

        const result = tidy(() => {
            // Compute probabilities for each class
            const probability = this.hypothesis(X, resolvedTheta);
            // Convert probabilities to class indices
            return this.probabilityToClassIndex(probability);
        });

        return result;
    }

    predictWithMetadata(X: Tensor2D, theta?: Tensor2D): PredictionMetadata {
        const resolvedTheta = theta ?? this.theta;

        assertModelTrained(resolvedTheta);

        const [probabilities, predictions] = tidy(() => {
            // Compute probabilities for each class
            const probability = this.hypothesis(X, resolvedTheta);
            // Convert probabilities to class indices
            const predictions = this.probabilityToClassIndex(probability) as Tensor2D;

            return [probability, predictions];
        });

        return {
            type: 'classification',
            predictions,
            probabilities,
            dispose() {
                predictions.dispose();
                probabilities.dispose();
            },
        };
    }

    usesOneHotLabels(): boolean {
        return true;
    }

    protected hypothesis(features: Tensor2D, theta: Tensor2D, asLogits = false): Tensor2D {
        // Softmax function for logistic regression
        const softmax = (z: Tensor2D): Tensor2D => {
            return z.softmax(); // exp(z) / sum(exp(z))
        };

        return tidy(() => {
            const z = this.addBiasTerm(features).matMul(theta) as Tensor2D;
            return asLogits ? z : softmax(z);
        });
    }

    protected probabilityToClassIndex(probability: Tensor2D): Tensor2D {
        return tidy(() => {
            // Find the indices of the maximum probabilities
            const maxIndices = probability.argMax(1);

            return maxIndices.reshape([-1, 1]) as Tensor2D;
        });
    }
}
