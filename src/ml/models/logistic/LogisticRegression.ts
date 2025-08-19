import { tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { BaseEstimator } from '../base/BaseEstimator';

export class LogisticRegression extends BaseEstimator {
    async train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D> {
        const asLogits = this.lossFunc.usesLogits?.();

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

        const theta = await this.optimizer.optimize({
            X,
            y,
            lossFunction,
            gradientFunction,
        });
        this.theta = theta; // Stack all thetas into a single tensor

        return this.theta;
    }

    predict(X: Tensor2D, theta?: Tensor2D): Tensor2D {
        if (!(theta ?? this.theta)) {
            throw new Error('Model has not been trained yet. Please call train() first.');
        }

        const result = tidy(() => {
            const probability = this.hypothesis(X, theta ?? this.theta!);
            const predictions = this.probabilityToClassIndex(probability);

            return predictions as Tensor2D;
        });

        // Dispose of the data to free memory
        X.dispose();

        return result;
    }

    evaluate(X: Tensor2D, y: Tensor2D, theta?: Tensor2D): [Tensor2D, Tensor2D, Scalar] {
        if (!(theta ?? this.theta)) {
            throw new Error('Model has not been trained yet. Please call train() first.');
        }

        const result = tidy(() => {
            const probability = this.hypothesis(X, theta ?? this.theta!);

            // Compute default loss using the loss function
            const loss = this.lossFunc.compute(y, probability);

            const yPred = this.probabilityToClassIndex(probability);

            return [yPred, probability, loss] as [Tensor2D, Tensor2D, Scalar];
        });

        return result;
    }

    protected hypothesis(features: Tensor2D, theta: Tensor2D, asLogits = false): Tensor2D {
        const sigmoid = (z: Tensor2D): Tensor2D => {
            return z.sigmoid(); //  1 / (1 + np.exp(-z))
        };

        return tidy(() => {
            const z = this.addBiasTerm(features).matMul(theta) as Tensor2D;
            return asLogits ? z : sigmoid(z);
        });
    }

    protected probabilityToClassIndex(probability: Tensor2D): Tensor2D {
        // Compare probabilities to 0.5 and set to 1 if >= 0.5, otherwise 0
        return tidy(() => probability.greaterEqual(0.5).cast('float32'));
    }
}
