import { concat, tidy, type Scalar, type Tensor2D } from '@tensorflow/tfjs';
import { LogisticRegressor } from './LogisticRegressor';
import type { PredictionMetadata } from '../../types';
import { assertModelTrained } from '../../utils';

export class OneVsRestLogisticRegressor extends LogisticRegressor {
    async train(X: Tensor2D, y: Tensor2D): Promise<Tensor2D> {
        const numFeatures = X.shape[1];
        const asLogits = this.lossFunc.usesLogits?.();

        const initTheta = this.thetaInitializer([numFeatures, 1]);

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
                    initTheta,
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

        initTheta.dispose();
        thetas.forEach((theta) => theta.dispose());

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

    private *classesDataIterator(
        X: Tensor2D,
        y: Tensor2D,
    ): IterableIterator<[number, [Tensor2D, Tensor2D]]> {
        const labels = y;
        const uniqueLabels = Array.from(new Set(labels.flatten().arraySync())); // WebGPU does not yet support the unique() function
        const numClasses = uniqueLabels.length;

        for (let labelIndex = 0; labelIndex < numClasses; labelIndex++) {
            const features = X.clone();
            const currentLabel = uniqueLabels[labelIndex];
            const currentLabels = labels.equal(currentLabel).cast('int32') as Tensor2D;

            yield [currentLabel, [features, currentLabels]];
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
