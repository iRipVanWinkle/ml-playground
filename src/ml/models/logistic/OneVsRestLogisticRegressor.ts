import { concat, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { LogisticRegressor } from './LogisticRegressor';

export class OneVsRestLogisticRegressor extends LogisticRegressor {
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

        const thetasPromise = [];
        for (const [label, [features, currentLabels]] of this.classesDataIterator(X, y)) {
            // Optimize theta using the provided optimizer
            const thetaPromise = (async () => {
                const theta = await this.optimizer.optimize({
                    X: features,
                    y: currentLabels,
                    lossFunction,
                    gradientFunction,
                    threadId: label,
                });

                // Dispose to free memory
                features.dispose();
                currentLabels.dispose();

                return theta;
            })();

            thetasPromise.push(thetaPromise); // Store the optimized theta for the current class
        }
        const thetas = await Promise.all(thetasPromise); // Wait for all theta optimizations to complete

        this.theta = concat(thetas, 1) as Tensor2D; // Stack all thetas into a single tensor

        thetas.forEach((theta) => theta.dispose());

        return this.theta;
    }

    predict(X: Tensor2D, theta?: Tensor2D): Tensor2D {
        if (!(theta ?? this.theta)) {
            throw new Error('Model has not been trained yet. Please call train() first.');
        }

        const result = tidy(() => {
            // Compute probabilities for each class
            const probability = this.hypothesis(X, theta ?? this.theta!);

            return this.probabilityToClassIndex(probability);
        });

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

    private *classesDataIterator(
        X: Tensor2D,
        y: Tensor2D,
    ): IterableIterator<[number, [Tensor2D, Tensor2D]]> {
        const labels = y;
        const uniqueLabels = tidy(() => labels.unique().values);
        const numClasses = uniqueLabels.shape[0];

        for (let labelIndex = 0; labelIndex < numClasses; labelIndex++) {
            const features = X.clone();
            const currentLabel = uniqueLabels.slice([labelIndex], [1]);
            const currentLabels = labels.equal(currentLabel).cast('int32') as Tensor2D;

            const currentLabelValue = currentLabel.dataSync()[0]; // Get the label value for the current class

            currentLabel.dispose();

            yield [currentLabelValue, [features, currentLabels]];
        }
    }

    protected probabilityToClassIndex(probability: Tensor2D): Tensor2D {
        return tidy(() => {
            // Find the indices of the maximum probabilities
            const maxIndices = probability.argMax(1);

            return maxIndices.reshape([-1, 1]) as Tensor2D;
        });
    }
}
