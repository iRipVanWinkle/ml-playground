import { concat, Tensor, type Tensor2D } from '@tensorflow/tfjs';
import type { ModelRepresentation } from '@/ml/types';

function fixLength(matrix: number[][]): number[][] {
    const minLength = Math.min(...matrix.map((m) => m.length));
    return matrix.map((m) => m.slice(0, minLength));
}

function isTensorParameter(param: ModelRepresentation): param is Tensor2D {
    return param instanceof Tensor && param.rank === 2;
}

export class TrainingSession {
    private readonly thetaHistory: Tensor2D[] = [];
    private readonly lossHistory: number[][] = [];
    private readonly iterationCounts: number[] = [];

    constructor({ numThreads }: { numThreads: number }) {
        this.lossHistory = Array.from({ length: numThreads }, () => []);
        this.iterationCounts = Array.from({ length: numThreads }, () => 0);
    }

    updateIteration(
        threadId: number,
        iteration: number,
        theta: ModelRepresentation,
        loss: number,
    ): void {
        if (isTensorParameter(theta) && this.thetaHistory[threadId] === undefined) {
            this.thetaHistory[threadId] = theta;
        }

        this.lossHistory[threadId].push(loss);
        this.iterationCounts[threadId] = iteration + 1;
    }

    getIterations(): number[] {
        return [...this.iterationCounts];
    }

    getCombinedTheta(): Tensor2D {
        return concat(this.thetaHistory.filter(Boolean), 1);
    }

    getFormattedLossHistory(): number[][] {
        return fixLength(this.lossHistory);
    }

    dispose(): void {
        this.thetaHistory.forEach((theta) => theta?.dispose());
    }
}
