import { tensor2d, type Tensor2D } from '@tensorflow/tfjs';
import type { Dataset } from '../types';

export class DatasetManager {
    private trainX: Tensor2D;
    private trainY: Tensor2D;
    private testX?: Tensor2D;
    private testY?: Tensor2D;
    private predictionX?: Tensor2D;

    private numClasses: number;

    constructor(data: Dataset) {
        this.trainX = tensor2d(data.trainInputFeatures);
        this.trainY = tensor2d(data.trainTargetLabels);

        if (data.testInputFeatures?.length) {
            this.testX = tensor2d(data.testInputFeatures);
            this.testY = tensor2d(data.testTargetLabels);
        }

        if (data.predictionInputFeatures?.length) {
            this.predictionX = tensor2d(data.predictionInputFeatures);
        }

        this.numClasses = data.categories?.length ?? 0;
    }

    getTrainingData(): { X: Tensor2D; y: Tensor2D } {
        return { X: this.trainX, y: this.trainY };
    }

    getTestData(): { X: Tensor2D; y: Tensor2D } | undefined {
        if (this.testX && this.testY) {
            return { X: this.testX, y: this.testY };
        }
        return undefined;
    }

    getPredictionData(): Tensor2D | undefined {
        return this.predictionX;
    }

    getNumClasses(): number {
        return this.numClasses;
    }

    dispose(): void {
        this.trainX.dispose();
        this.trainY.dispose();
        this.testX?.dispose();
        this.testY?.dispose();
        this.predictionX?.dispose();
    }
}
