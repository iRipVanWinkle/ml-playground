import { type Scalar, type Tensor2D, tidy } from '@tensorflow/tfjs';
import { BaseEstimator } from '../base/BaseEstimator';
import { assertThetaTrained } from '../../utils';

export class LinearRegressor extends BaseEstimator {
    async train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D> {
        const numFeatures = X.shape[1];

        const initTheta = this.thetaInitializer([numFeatures, 1]);

        // Define the loss function
        const lossFunction = (X: Tensor2D, y: Tensor2D, theta: Tensor2D): Scalar => {
            // Compute the predictions using the hypothesis function
            const yPred = this.hypothesis(X, theta);
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
        assertThetaTrained(theta ?? this.theta);

        const result = this.hypothesis(X, theta ?? this.theta!);

        return result;
    }

    evaluate(X: Tensor2D, y: Tensor2D, theta?: Tensor2D): [Tensor2D, Tensor2D, Scalar] {
        assertThetaTrained(theta ?? this.theta);

        const result = tidy(() => {
            const yPred = this.hypothesis(X, theta ?? this.theta!);

            const loss = this.lossFunc.compute(y, yPred);

            return [yPred, yPred, loss] as [Tensor2D, Tensor2D, Scalar];
        });

        return result;
    }

    private hypothesis(features: Tensor2D, theta: Tensor2D): Tensor2D {
        return tidy(() => this.addBiasTerm(features).matMul(theta) as Tensor2D);
    }
}
